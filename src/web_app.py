import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from .generate_dataset import DATASET_PATH, generate_dataset
except ImportError:
    from generate_dataset import DATASET_PATH, generate_dataset


HOST = "127.0.0.1"
PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
USERS_PATH = PROJECT_ROOT / "data" / "users.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "risk_history.json"

NUMERIC_FEATURES = [
    "seniority_years",
    "total_weekly_hours",
    "overtime_hours",
    "daily_meeting_volume",
    "days_since_last_vacation",
    "sick_leave_days_6m",
    "self_reported_stress",
    "motivation_level",
]
CATEGORICAL_FEATURES = ["department", "role"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RECOMMENDATIONS = {
    "Low": [
        "Maintain regular workload monitoring and balanced meeting schedules.",
        "Encourage the employee to keep taking vacations at healthy intervals.",
        "Continue periodic pulse surveys to detect early changes in stress or motivation.",
    ],
    "Medium": [
        "Schedule a check-in with the employee or manager to understand current pressure points.",
        "Review overtime and meeting volume for the next two to four weeks.",
        "Encourage short recovery periods, workload prioritization, and use of available vacation days.",
    ],
    "High": [
        "Prioritize an immediate confidential conversation focused on support, not blame.",
        "Reduce overtime and non-essential meetings as soon as operationally possible.",
        "Consider HR, occupational health, or wellbeing support, respecting privacy and consent.",
    ],
}
RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3}


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_users():
    return read_json(USERS_PATH, [])


def save_users(users):
    write_json(USERS_PATH, users)


def load_history():
    return read_json(HISTORY_PATH, [])


def save_history(history):
    write_json(HISTORY_PATH, history)


def initials(first_name, last_name):
    return f"{first_name.strip()[0]}{last_name.strip()[0]}".upper()


def next_user_id(first_name, last_name):
    prefix = initials(first_name, last_name)
    users = load_users()
    return f"{prefix}{len(users) + 1:02d}"


def public_user(user):
    return {
        "employee_id": user["employee_id"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "email": user["email"],
    }


def create_user(payload):
    first_name = str(payload.get("first_name", "")).strip()
    last_name = str(payload.get("last_name", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    pin = str(payload.get("pin", "")).strip()

    if not first_name or not last_name:
        raise ValueError("First and last name are required.")
    if "@" not in email or "." not in email:
        raise ValueError("A valid email is required.")
    if not pin.isdigit() or len(pin) != 6:
        raise ValueError("PIN must contain exactly six digits.")

    users = load_users()
    if any(user["email"].lower() == email for user in users):
        raise ValueError("This email is already registered.")

    user = {
        "employee_id": next_user_id(first_name, last_name),
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "pin": pin,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    users.append(user)
    save_users(users)
    return public_user(user)


def authenticate(payload):
    employee_id = str(payload.get("employee_id", "")).strip().upper()
    pin = str(payload.get("pin", "")).strip()
    for user in load_users():
        if user["employee_id"].upper() == employee_id and user["pin"] == pin:
            return public_user(user)
    raise ValueError("Invalid ID or PIN.")


def trend_for(records):
    if len(records) < 2:
        return "Not enough history yet"
    previous = RISK_ORDER[records[-2]["risk"]]
    current = RISK_ORDER[records[-1]["risk"]]
    if current < previous:
        return "Improving"
    if current > previous:
        return "Worsening"
    return "Stable"


def history_for_employee(employee_id):
    employee_id = employee_id.strip().upper()
    records = [
        record for record in load_history()
        if record["employee_id"].upper() == employee_id
    ]
    records.sort(key=lambda record: record["timestamp"])
    return {"records": records, "trend": trend_for(records)}


def add_history_record(payload, risk, probabilities):
    history = load_history()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "employee_id": payload["employee_id"],
        "employee_name": payload["employee_name"],
        "age": payload["age"],
        "department": payload["department"],
        "role": payload["role"],
        "risk": risk,
        "probabilities": probabilities,
        "inputs": {feature: payload[feature] for feature in FEATURES},
    }
    history.append(record)
    save_history(history)
    return record


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def load_or_create_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    return dataset


def train_model():
    dataset = load_or_create_dataset()
    x = dataset[FEATURES]
    y = dataset["burnout_risk"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    models = build_models()
    best_name = None
    best_score = -1

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=["Low", "Medium", "High"],
            average="macro",
            zero_division=0,
        )
        if f1 > best_score:
            best_name = model_name
            best_score = f1

    best_model = build_models()[best_name]
    best_model.fit(x, y)
    return best_model, best_name, best_score


MODEL, MODEL_NAME, MODEL_F1 = train_model()


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Employee Burnout Risk Prediction</title>
  <style>
    :root {
      font-family: Arial, Helvetica, sans-serif;
      --blue: #1f5d8f;
      --red: #b42318;
      --amber: #b7791f;
      --green: #237a3b;
      --ink: #172033;
      --muted: #5c667a;
      --line: #d8dee9;
      --surface: #f6f8fb;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--surface); color: var(--ink); }
    header {
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
      background: #fff; border-bottom: 1px solid var(--line); padding: 22px 32px;
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px 32px; }
    .grid { display: grid; grid-template-columns: minmax(320px, 1.2fr) minmax(280px, 0.8fr); gap: 24px; }
    section { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 22px; }
    h2 { margin: 0 0 18px; font-size: 18px; }
    h3 { margin: 18px 0 10px; font-size: 16px; }
    form, fieldset { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    fieldset { grid-column: 1 / -1; border: 0; border-top: 1px solid var(--line); margin: 4px 0 0; padding: 18px 0 0; }
    fieldset:first-child { border-top: 0; padding-top: 0; margin-top: 0; }
    legend { grid-column: 1 / -1; font-size: 15px; font-weight: 800; padding: 0 0 2px; }
    label { display: grid; gap: 7px; color: var(--muted); font-size: 13px; font-weight: 700; }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px; font: inherit; color: var(--ink); background: #fff; }
    button { border: 0; border-radius: 6px; background: var(--blue); color: white; font: inherit; font-weight: 700; padding: 12px 16px; cursor: pointer; }
    button:hover { background: #17486f; }
    .secondary { background: #fff; color: var(--blue); border: 1px solid var(--blue); }
    .secondary:hover { background: #edf5fc; }
    .auth { max-width: 760px; margin: 0 auto; }
    .auth-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px; }
    .auth-tabs button.active { background: var(--blue); color: #fff; }
    .auth-tabs button { background: #fff; color: var(--blue); border: 1px solid var(--blue); }
    .actions { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; grid-column: 1 / -1; }
    .hidden { display: none !important; }
    .top-tools { display: flex; align-items: center; gap: 12px; }
    .user-chip { color: var(--muted); font-weight: 700; }
    .icon-btn { width: 42px; height: 42px; padding: 0; font-size: 20px; }
    .search-panel { margin-bottom: 24px; }
    .search-panel form { grid-template-columns: minmax(220px, 1fr) auto; align-items: end; }
    .badge { display: inline-flex; align-items: center; justify-content: center; min-height: 54px; border-radius: 8px; color: white; font-size: 26px; font-weight: 800; background: var(--amber); }
    .badge.low { background: var(--green); }
    .badge.medium { background: var(--amber); }
    .badge.high { background: var(--red); }
    .probabilities, .result { display: grid; gap: 12px; }
    .bar-label { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); }
    .bar { height: 12px; border-radius: 999px; background: #e8edf5; overflow: hidden; }
    .bar span { display: block; height: 100%; background: var(--blue); }
    .note, .message { color: var(--muted); line-height: 1.45; margin: 0; }
    .message.error { color: var(--red); font-weight: 700; }
    .report { grid-column: 1 / -1; }
    .report-head, .history-head { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 16px; }
    dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0; }
    dt { color: var(--muted); font-size: 12px; font-weight: 700; }
    dd { margin: 4px 0 0; font-weight: 700; }
    .suggestions { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .suggestion-group { border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .suggestion-group.active { border-width: 2px; border-color: var(--blue); background: #f2f8fd; }
    .suggestion-group h4 { margin: 0 0 10px; font-size: 15px; }
    .suggestion-group ul { margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.4; }
    .trend { border-radius: 999px; padding: 8px 12px; background: #eef3f8; font-weight: 800; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    @media print {
      header, .auth, .search-panel, main > .grid > section:first-child, main > .grid > section:nth-child(2) { display: none; }
      body { background: #fff; } main { padding: 0; } .report { border: 0; padding: 0; }
      .suggestions { grid-template-columns: 1fr; }
    }
    @media (max-width: 820px) {
      header { padding: 18px 16px; } main { padding: 16px; }
      .grid, form, fieldset, .actions, .search-panel form, .suggestions { grid-template-columns: 1fr; }
      dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Employee Burnout Risk Prediction</h1>
    <div id="top-tools" class="top-tools hidden">
      <span id="user-chip" class="user-chip"></span>
      <button id="search-toggle" class="icon-btn" title="Search history">⌕</button>
      <button id="logout-button" class="secondary">Logout</button>
    </div>
  </header>

  <main>
    <section id="auth-section" class="auth">
      <div class="auth-tabs">
        <button id="show-register" class="active">Register</button>
        <button id="show-login" type="button">Login</button>
      </div>

      <form id="register-form">
        <label>First name
          <input name="first_name" type="text" placeholder="e.g. Joao" required>
        </label>
        <label>Last name
          <input name="last_name" type="text" placeholder="e.g. Barbosa" required>
        </label>
        <label>Email
          <input name="email" type="email" placeholder="melissa@email.com" required>
        </label>
        <label>PIN
          <input name="pin" type="password" pattern="[0-9]{6}" maxlength="6" inputmode="numeric" placeholder="6 digits" required>
        </label>
        <button type="submit">Register and enter</button>
        <p id="register-message" class="message"></p>
      </form>

      <form id="login-form" class="hidden">
        <label>Employee ID
          <input name="employee_id" type="text" placeholder="e.g. JB01" required>
        </label>
        <label>PIN
          <input name="pin" type="password" pattern="[0-9]{6}" maxlength="6" inputmode="numeric" placeholder="6 digits" required>
        </label>
        <button type="submit">Login</button>
        <p id="login-message" class="message"></p>
      </form>
    </section>

    <div id="app-section" class="hidden">
      <section id="search-panel" class="search-panel hidden">
        <h2>Search employee history</h2>
        <p class="note">Search by generated Employee ID.</p>
        <form id="history-form">
          <label>Employee ID
            <input name="history_employee_id" type="text" placeholder="e.g. JB01" required>
          </label>
          <button type="submit">Search history</button>
        </form>
        <p id="history-message" class="message"></p>
      </section>

      <section class="hidden" id="history-panel">
        <div class="history-head">
          <div>
            <h2>Risk history</h2>
            <p class="note" id="history-subtitle"></p>
          </div>
          <div class="trend" id="history-trend"></div>
        </div>
        <div id="history-content"></div>
      </section>

      <div class="grid">
        <section>
          <h2>Employee profile and work indicators</h2>
          <form id="prediction-form">
            <fieldset>
              <legend>Personal and job profile</legend>
              <label>Age
                <input name="age" type="number" min="18" max="70" step="1" placeholder="e.g. 22" required>
              </label>
              <label>Department
                <select name="department" required>
                  <option value="" selected disabled>Select department</option>
                  <option>IT</option><option>Sales</option><option>HR</option>
                  <option>Finance</option><option>Operations</option><option>Marketing</option>
                </select>
              </label>
              <label>Role
                <select name="role" required>
                  <option value="" selected disabled>Select role</option>
                  <option>Junior</option><option>Mid</option><option>Senior</option>
                  <option>Lead</option><option>Manager</option>
                </select>
              </label>
              <label>Seniority years
                <input name="seniority_years" type="number" min="0" max="20" step="0.1" placeholder="e.g. 4" required>
              </label>
            </fieldset>
            <fieldset>
              <legend>Activity indicators</legend>
              <label>Total weekly hours
                <input name="total_weekly_hours" type="number" min="32" max="70" step="0.1" placeholder="e.g. 40" required>
              </label>
              <label>Overtime hours
                <input name="overtime_hours" type="number" min="0" max="25" step="0.1" placeholder="e.g. 5" required>
              </label>
              <label>Daily meeting volume
                <input name="daily_meeting_volume" type="number" min="0" max="7" step="0.1" placeholder="e.g. 2.5" required>
              </label>
            </fieldset>
            <fieldset>
              <legend>Well-being indicators</legend>
              <label>Days since last vacation
                <input name="days_since_last_vacation" type="number" min="2" max="420" step="1" placeholder="e.g. 90" required>
              </label>
              <label>Sick leave days, last 6 months
                <input name="sick_leave_days_6m" type="number" min="0" max="22" step="1" placeholder="e.g. 2" required>
              </label>
            </fieldset>
            <fieldset>
              <legend>Pulse survey</legend>
              <label>Self-reported stress
                <input name="self_reported_stress" type="number" min="1" max="5" step="1" placeholder="1 to 5" required>
              </label>
              <label>Motivation level
                <input name="motivation_level" type="number" min="1" max="5" step="1" placeholder="1 to 5" required>
              </label>
            </fieldset>
            <div class="actions">
              <button type="submit">Predict burnout risk</button>
              <button type="reset" id="clear-form" class="secondary">Clear form</button>
              <button type="button" id="print-report" class="secondary">Print report</button>
            </div>
          </form>
        </section>

        <section class="result hidden" id="prediction-panel">
          <h2>Prediction</h2>
          <div id="risk-badge" class="badge medium"></div>
          <div id="probabilities" class="probabilities"></div>
          <p class="note">The prediction is generated by the best evaluated model: <span id="model-name">selected model</span>.</p>
        </section>

        <section class="report hidden" id="report">
          <div class="report-head">
            <div>
              <h2>Burnout risk report</h2>
              <p class="note" id="report-subtitle"></p>
            </div>
            <div id="report-risk" class="badge medium"></div>
          </div>
          <h3>Employee information</h3>
          <dl id="employee-summary"></dl>
          <h3>Model probabilities</h3>
          <div id="report-probabilities" class="probabilities"></div>
          <h3>Recommendations by risk level</h3>
          <div id="recommendations" class="suggestions"></div>
        </section>
      </div>
    </div>
  </main>

  <script>
    let currentUser = null;
    const authSection = document.querySelector("#auth-section");
    const appSection = document.querySelector("#app-section");
    const topTools = document.querySelector("#top-tools");
    const userChip = document.querySelector("#user-chip");
    const registerForm = document.querySelector("#register-form");
    const loginForm = document.querySelector("#login-form");
    const showRegister = document.querySelector("#show-register");
    const showLogin = document.querySelector("#show-login");
    const registerMessage = document.querySelector("#register-message");
    const loginMessage = document.querySelector("#login-message");
    const searchToggle = document.querySelector("#search-toggle");
    const searchPanel = document.querySelector("#search-panel");
    const historyForm = document.querySelector("#history-form");
    const form = document.querySelector("#prediction-form");
    const predictionPanel = document.querySelector("#prediction-panel");
    const report = document.querySelector("#report");
    const historyPanel = document.querySelector("#history-panel");
    const historySubtitle = document.querySelector("#history-subtitle");
    const historyTrend = document.querySelector("#history-trend");
    const historyContent = document.querySelector("#history-content");
    const historyMessage = document.querySelector("#history-message");
    const badge = document.querySelector("#risk-badge");
    const probabilities = document.querySelector("#probabilities");
    const reportRisk = document.querySelector("#report-risk");
    const reportSubtitle = document.querySelector("#report-subtitle");
    const employeeSummary = document.querySelector("#employee-summary");
    const reportProbabilities = document.querySelector("#report-probabilities");
    const recommendations = document.querySelector("#recommendations");
    const modelName = document.querySelector("#model-name");

    const recommendationText = {
      Low: ["Maintain regular workload monitoring and balanced meeting schedules.", "Encourage the employee to keep taking vacations at healthy intervals.", "Continue periodic pulse surveys to detect early changes in stress or motivation."],
      Medium: ["Schedule a check-in with the employee or manager to understand current pressure points.", "Review overtime and meeting volume for the next two to four weeks.", "Encourage short recovery periods, workload prioritization, and use of available vacation days."],
      High: ["Prioritize an immediate confidential conversation focused on support, not blame.", "Reduce overtime and non-essential meetings as soon as operationally possible.", "Consider HR, occupational health, or wellbeing support, respecting privacy and consent."]
    };

    function toPayload(formElement) {
      const payload = {};
      for (const [key, value] of new FormData(formElement).entries()) {
        payload[key] = Number.isNaN(Number(value)) || value.trim() === "" ? value : Number(value);
      }
      return payload;
    }

    async function postJson(path, payload) {
      const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed.");
      return data;
    }

    function enterApp(user) {
      currentUser = user;
      authSection.classList.add("hidden");
      appSection.classList.remove("hidden");
      topTools.classList.remove("hidden");
      userChip.textContent = `${user.first_name} ${user.last_name} (${user.employee_id})`;
    }

    showRegister.addEventListener("click", () => {
      registerForm.classList.remove("hidden");
      loginForm.classList.add("hidden");
      showRegister.classList.add("active");
      showLogin.classList.remove("active");
    });

    showLogin.addEventListener("click", () => {
      loginForm.classList.remove("hidden");
      registerForm.classList.add("hidden");
      showLogin.classList.add("active");
      showRegister.classList.remove("active");
    });

    registerForm.addEventListener("submit", async event => {
      event.preventDefault();
      registerMessage.textContent = "";
      registerMessage.className = "message";
      try {
        const data = await postJson("/register", toPayload(registerForm));
        registerMessage.textContent = `Registered with ID ${data.user.employee_id}`;
        enterApp(data.user);
      } catch (error) {
        registerMessage.textContent = error.message;
        registerMessage.className = "message error";
      }
    });

    loginForm.addEventListener("submit", async event => {
      event.preventDefault();
      loginMessage.textContent = "";
      loginMessage.className = "message";
      try {
        const data = await postJson("/login", toPayload(loginForm));
        enterApp(data.user);
      } catch (error) {
        loginMessage.textContent = error.message;
        loginMessage.className = "message error";
      }
    });

    function renderBars(container, data) {
      container.innerHTML = "";
      for (const [label, probability] of Object.entries(data.probabilities)) {
        const percent = Math.round(probability * 100);
        const row = document.createElement("div");
        row.innerHTML = `<div class="bar-label"><strong>${label}</strong><span>${percent}%</span></div><div class="bar"><span style="width: ${percent}%"></span></div>`;
        container.appendChild(row);
      }
    }

    function renderRecommendations(activeRisk) {
      recommendations.innerHTML = "";
      for (const [risk, items] of Object.entries(recommendationText)) {
        const group = document.createElement("div");
        group.className = `suggestion-group ${risk === activeRisk ? "active" : ""}`;
        group.innerHTML = `<h4>${risk} risk</h4><ul>${items.map(item => `<li>${item}</li>`).join("")}</ul>`;
        recommendations.appendChild(group);
      }
    }

    function renderEmployeeSummary(payload) {
      const entries = [["ID", currentUser.employee_id], ["Name", `${currentUser.first_name} ${currentUser.last_name}`], ["Age", payload.age], ["Department", payload.department], ["Role", payload.role], ["Seniority", `${payload.seniority_years} years`], ["Weekly hours", payload.total_weekly_hours], ["Overtime", `${payload.overtime_hours} hours`]];
      employeeSummary.innerHTML = entries.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
    }

    function renderPrediction(data, payload) {
      predictionPanel.classList.remove("hidden");
      report.classList.remove("hidden");
      badge.textContent = data.risk;
      badge.className = `badge ${data.risk.toLowerCase()}`;
      reportRisk.textContent = data.risk;
      reportRisk.className = `badge ${data.risk.toLowerCase()}`;
      reportSubtitle.textContent = `Report for ${currentUser.first_name} ${currentUser.last_name} (${currentUser.employee_id}).`;
      modelName.textContent = data.model_name;
      renderBars(probabilities, data);
      renderBars(reportProbabilities, data);
      renderEmployeeSummary(payload);
      renderRecommendations(data.risk);
    }

    async function fetchHistory(employeeId) {
      const response = await fetch(`/history?employee_id=${encodeURIComponent(employeeId)}`);
      return response.json();
    }

    function renderHistory(data, employeeId) {
      historyPanel.classList.remove("hidden");
      historySubtitle.textContent = `Employee ID: ${employeeId}`;
      historyTrend.textContent = `Trend: ${data.trend}`;
      if (data.records.length === 0) {
        historyContent.innerHTML = "<p class='note'>No previous reports found for this ID.</p>";
        historyPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      const rows = data.records.map((record, index) => {
        const date = new Date(record.timestamp).toLocaleString();
        return `<tr><td>${index + 1}</td><td>${date}</td><td>${record.employee_name}</td><td>${record.risk}</td><td>${Math.round(record.probabilities.Low * 100)}%</td><td>${Math.round(record.probabilities.Medium * 100)}%</td><td>${Math.round(record.probabilities.High * 100)}%</td></tr>`;
      }).join("");
      historyContent.innerHTML = `<table><thead><tr><th>#</th><th>Date</th><th>Name</th><th>Risk</th><th>Low</th><th>Medium</th><th>High</th></tr></thead><tbody>${rows}</tbody></table>`;
      historyPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    form.addEventListener("submit", async event => {
      event.preventDefault();
      const payload = { ...toPayload(form), employee_id: currentUser.employee_id, employee_name: `${currentUser.first_name} ${currentUser.last_name}` };
      const data = await postJson("/predict", payload);
      renderPrediction(data, payload);
      renderHistory(await fetchHistory(currentUser.employee_id), currentUser.employee_id);
    });

    document.querySelector("#clear-form").addEventListener("click", () => {
      predictionPanel.classList.add("hidden");
      report.classList.add("hidden");
      probabilities.innerHTML = "";
      reportProbabilities.innerHTML = "";
    });

    document.querySelector("#print-report").addEventListener("click", () => {
      if (report.classList.contains("hidden")) return;
      window.print();
    });

    searchToggle.addEventListener("click", () => searchPanel.classList.toggle("hidden"));
    historyForm.addEventListener("submit", async event => {
      event.preventDefault();
      const employeeId = new FormData(historyForm).get("history_employee_id").trim();
      historyMessage.textContent = "Searching...";
      historyMessage.className = "message";
      try {
        const data = await fetchHistory(employeeId);
        renderHistory(data, employeeId);
        historyMessage.textContent = data.records.length === 0 ? "No history found." : `${data.records.length} report(s) found.`;
      } catch (error) {
        historyMessage.textContent = "Could not load history.";
        historyMessage.className = "message error";
      }
    });

    document.querySelector("#logout-button").addEventListener("click", () => {
      currentUser = null;
      authSection.classList.remove("hidden");
      appSection.classList.add("hidden");
      topTools.classList.add("hidden");
      form.reset();
    });
  </script>
</body>
</html>
"""


class BurnoutRiskHandler(BaseHTTPRequestHandler):
    def _send(self, status_code, body, content_type):
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json_response(self, status_code, payload):
        self._send(status_code, json.dumps(payload), "application/json; charset=utf-8")

    def _read_payload(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if parsed_url.path == "/history":
            employee_id = parse_qs(parsed_url.query).get("employee_id", [""])[0]
            self._json_response(200, history_for_employee(employee_id))
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        try:
            payload = self._read_payload()
            path = urlparse(self.path).path
            if path == "/register":
                self._json_response(201, {"user": create_user(payload)})
                return
            if path == "/login":
                self._json_response(200, {"user": authenticate(payload)})
                return
            if path == "/predict":
                model_payload = {feature: payload[feature] for feature in FEATURES}
                row = pd.DataFrame([model_payload], columns=FEATURES)
                risk = MODEL.predict(row)[0]
                probabilities = MODEL.predict_proba(row)[0]
                classes = list(MODEL.classes_)
                probability_map = {
                    label: round(float(probabilities[classes.index(label)]), 4)
                    for label in ["Low", "Medium", "High"]
                }
                record = add_history_record(payload, risk, probability_map)
                self._json_response(
                    200,
                    {
                        "risk": risk,
                        "model_name": MODEL_NAME,
                        "model_f1_macro": round(float(MODEL_F1), 4),
                        "probabilities": probability_map,
                        "record": record,
                    },
                )
                return
            self._send(404, "Not found", "text/plain; charset=utf-8")
        except Exception as error:
            self._json_response(400, {"error": str(error)})

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), BurnoutRiskHandler)
    print(f"Web app running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
