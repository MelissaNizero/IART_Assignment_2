import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .generate_dataset import DATASET_PATH, generate_dataset
except ImportError:
    from generate_dataset import DATASET_PATH, generate_dataset


HOST = "127.0.0.1"
PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


MODEL = train_model()


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Employee Burnout Risk Prediction</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      --blue: #1f5d8f;
      --teal: #14746f;
      --red: #b42318;
      --amber: #b7791f;
      --green: #237a3b;
      --ink: #172033;
      --muted: #5c667a;
      --line: #d8dee9;
      --surface: #f6f8fb;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--surface);
      color: var(--ink);
    }

    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 22px 32px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }

    main {
      display: grid;
      grid-template-columns: minmax(320px, 1.2fr) minmax(280px, 0.8fr);
      gap: 24px;
      padding: 24px 32px;
      max-width: 1180px;
      margin: 0 auto;
    }

    section {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
    }

    h2 {
      margin: 0 0 18px;
      font-size: 18px;
    }

    form {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
    }

    button {
      grid-column: 1 / -1;
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      font: inherit;
      font-weight: 700;
      padding: 12px 16px;
      cursor: pointer;
    }

    button:hover { background: #17486f; }

    .result {
      display: grid;
      gap: 18px;
    }

    .hidden {
      display: none;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 54px;
      border-radius: 8px;
      color: white;
      font-size: 26px;
      font-weight: 800;
      background: var(--teal);
    }

    .badge.low { background: var(--green); }
    .badge.medium { background: var(--amber); }
    .badge.high { background: var(--red); }

    .probabilities {
      display: grid;
      gap: 12px;
    }

    .bar-row {
      display: grid;
      gap: 5px;
    }

    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: var(--muted);
    }

    .bar {
      height: 12px;
      border-radius: 999px;
      background: #e8edf5;
      overflow: hidden;
    }

    .bar span {
      display: block;
      height: 100%;
      width: 0;
      background: var(--blue);
      transition: width 160ms ease;
    }

    .note {
      color: var(--muted);
      line-height: 1.45;
      margin: 0;
    }

    .full {
      grid-column: 1 / -1;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      grid-column: 1 / -1;
    }

    .secondary {
      background: #ffffff;
      color: var(--blue);
      border: 1px solid var(--blue);
    }

    .secondary:hover {
      background: #edf5fc;
    }

    .report {
      grid-column: 1 / -1;
    }

    .report-head {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
      margin-bottom: 16px;
    }

    .report h3 {
      margin: 18px 0 10px;
      font-size: 16px;
    }

    .report dl {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 0;
    }

    .report dt {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .report dd {
      margin: 4px 0 0;
      font-weight: 700;
    }

    .suggestions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .suggestion-group {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }

    .suggestion-group.active {
      border-width: 2px;
      border-color: var(--blue);
      background: #f2f8fd;
    }

    .suggestion-group h4 {
      margin: 0 0 10px;
      font-size: 15px;
    }

    .suggestion-group ul {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.4;
    }

    @media print {
      body { background: #ffffff; }
      header, main > section:first-child, main > section:nth-child(2) {
        display: none;
      }
      main {
        display: block;
        max-width: none;
        padding: 0;
      }
      .report {
        border: 0;
        padding: 0;
      }
      .suggestions {
        grid-template-columns: 1fr;
      }
      .secondary {
        display: none;
      }
    }

    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; padding: 16px; }
      header { padding: 18px 16px; }
      form { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr; }
      .report dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .suggestions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Employee Burnout Risk Prediction</h1>
  </header>

  <main>
    <section>
      <h2>Employee profile and work indicators</h2>
      <form id="prediction-form">
        <label>Employee name
          <input name="employee_name" type="text" autocomplete="name" required>
        </label>

        <label>Age
          <input name="age" type="number" min="18" max="70" step="1" required>
        </label>

        <label>Department
          <select name="department">
            <option>IT</option>
            <option>Sales</option>
            <option>HR</option>
            <option>Finance</option>
            <option>Operations</option>
            <option>Marketing</option>
          </select>
        </label>

        <label>Role
          <select name="role">
            <option>Junior</option>
            <option>Mid</option>
            <option>Senior</option>
            <option>Lead</option>
            <option>Manager</option>
          </select>
        </label>

        <label>Seniority years
          <input name="seniority_years" type="number" min="0" max="20" step="0.1" required>
        </label>

        <label>Total weekly hours
          <input name="total_weekly_hours" type="number" min="32" max="70" step="0.1" required>
        </label>

        <label>Overtime hours
          <input name="overtime_hours" type="number" min="0" max="25" step="0.1" required>
        </label>

        <label>Daily meeting volume
          <input name="daily_meeting_volume" type="number" min="0" max="7" step="0.1" required>
        </label>

        <label>Days since last vacation
          <input name="days_since_last_vacation" type="number" min="2" max="420" step="1" required>
        </label>

        <label>Sick leave days, last 6 months
          <input name="sick_leave_days_6m" type="number" min="0" max="22" step="1" required>
        </label>

        <label>Self-reported stress
          <input name="self_reported_stress" type="number" min="1" max="5" step="1" required>
        </label>

        <label>Motivation level
          <input name="motivation_level" type="number" min="1" max="5" step="1" required>
        </label>

        <div class="actions">
          <button type="submit">Predict burnout risk</button>
          <button type="button" id="print-report" class="secondary">Print report</button>
        </div>
      </form>
    </section>

    <section class="result hidden" id="prediction-panel">
      <h2>Prediction</h2>
      <div id="risk-badge" class="badge medium"></div>
      <div id="probabilities" class="probabilities"></div>
      <p class="note">
        The prediction is generated by the Logistic Regression model trained on
        the synthetic dataset used in the empirical study.
      </p>
    </section>

    <section class="report hidden" id="report">
      <div class="report-head">
        <div>
          <h2>Burnout risk report</h2>
          <p class="note" id="report-subtitle">Generated after risk prediction.</p>
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
  </main>

  <script>
    const form = document.querySelector("#prediction-form");
    const predictionPanel = document.querySelector("#prediction-panel");
    const report = document.querySelector("#report");
    const badge = document.querySelector("#risk-badge");
    const probabilities = document.querySelector("#probabilities");
    const reportRisk = document.querySelector("#report-risk");
    const reportSubtitle = document.querySelector("#report-subtitle");
    const employeeSummary = document.querySelector("#employee-summary");
    const reportProbabilities = document.querySelector("#report-probabilities");
    const recommendations = document.querySelector("#recommendations");
    const printButton = document.querySelector("#print-report");

    const recommendationText = {
      Low: [
        "Maintain regular workload monitoring and balanced meeting schedules.",
        "Encourage the employee to keep taking vacations at healthy intervals.",
        "Continue periodic pulse surveys to detect early changes in stress or motivation."
      ],
      Medium: [
        "Schedule a check-in with the employee or manager to understand current pressure points.",
        "Review overtime and meeting volume for the next two to four weeks.",
        "Encourage short recovery periods, workload prioritization, and use of available vacation days."
      ],
      High: [
        "Prioritize an immediate confidential conversation focused on support, not blame.",
        "Reduce overtime and non-essential meetings as soon as operationally possible.",
        "Consider HR, occupational health, or wellbeing support, respecting privacy and consent."
      ]
    };

    function formDataToPayload() {
      const data = new FormData(form);
      const payload = {};
      for (const [key, value] of data.entries()) {
        payload[key] = Number.isNaN(Number(value)) || value.trim() === "" ? value : Number(value);
      }
      return payload;
    }

    function renderBars(container, data) {
      container.innerHTML = "";
      for (const [label, probability] of Object.entries(data.probabilities)) {
        const percent = Math.round(probability * 100);
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-label"><strong>${label}</strong><span>${percent}%</span></div>
          <div class="bar"><span style="width: ${percent}%"></span></div>
        `;
        container.appendChild(row);
      }
    }

    function renderEmployeeSummary(payload) {
      const entries = [
        ["Name", payload.employee_name || "Not provided"],
        ["Age", payload.age],
        ["Department", payload.department],
        ["Role", payload.role],
        ["Seniority", `${payload.seniority_years} years`],
        ["Weekly hours", payload.total_weekly_hours],
        ["Overtime", `${payload.overtime_hours} hours`],
        ["Days since vacation", payload.days_since_last_vacation]
      ];

      employeeSummary.innerHTML = entries.map(([label, value]) => `
        <div><dt>${label}</dt><dd>${value}</dd></div>
      `).join("");
    }

    function renderRecommendations(activeRisk) {
      recommendations.innerHTML = "";
      for (const [risk, items] of Object.entries(recommendationText)) {
        const group = document.createElement("div");
        group.className = `suggestion-group ${risk === activeRisk ? "active" : ""}`;
        group.innerHTML = `
          <h4>${risk} risk</h4>
          <ul>${items.map(item => `<li>${item}</li>`).join("")}</ul>
        `;
        recommendations.appendChild(group);
      }
    }

    function renderPrediction(data, payload) {
      const risk = data.risk;
      predictionPanel.classList.remove("hidden");
      report.classList.remove("hidden");
      badge.textContent = risk;
      badge.className = `badge ${risk.toLowerCase()}`;
      reportRisk.textContent = risk;
      reportRisk.className = `badge ${risk.toLowerCase()}`;
      reportSubtitle.textContent = `Report for ${payload.employee_name || "employee"} generated with the trained ML model.`;
      renderBars(probabilities, data);
      renderBars(reportProbabilities, data);
      renderEmployeeSummary(payload);
      renderRecommendations(risk);
    }

    async function predict() {
      const payload = formDataToPayload();
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderPrediction(await response.json(), payload);
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      predict();
    });

    printButton.addEventListener("click", async () => {
      if (!form.reportValidity()) {
        return;
      }
      await predict();
      window.print();
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

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/predict":
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        model_payload = {feature: payload[feature] for feature in FEATURES}
        row = pd.DataFrame([model_payload], columns=FEATURES)

        risk = MODEL.predict(row)[0]
        probabilities = MODEL.predict_proba(row)[0]
        classes = MODEL.classes_
        response = {
            "risk": risk,
            "probabilities": {
                label: round(float(probabilities[list(classes).index(label)]), 4)
                for label in ["Low", "Medium", "High"]
            },
        }
        self._send(200, json.dumps(response), "application/json; charset=utf-8")

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), BurnoutRiskHandler)
    print(f"Web app running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
