const API_BASE = "http://127.0.0.1:8000";  // your FastAPI server
const API_KEY = ""; // if you set API_KEY env var, put the same here as "x-api-key"

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("fraud-form");
  const resultDiv = document.getElementById("result");
  const rawDiv = document.getElementById("raw-response");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Read important fields from the form
    const income = parseFloat(document.getElementById("income").value);
    const customer_age = parseFloat(document.getElementById("customer_age").value);
    const credit_risk_score = parseFloat(document.getElementById("credit_risk_score").value);
    const velocity_24h = parseFloat(document.getElementById("velocity_24h").value);
    const device_fraud_count = parseFloat(document.getElementById("device_fraud_count").value);
    const name_email_similarity = parseFloat(document.getElementById("name_email_similarity").value);

    // Build full payload: ALL fields required by Transaction model
    const payload = {
      income,
      name_email_similarity,
      prev_address_months_count: 0.0,
      current_address_months_count: 0.0,
      customer_age,
      days_since_request: 0.0,
      intended_balcon_amount: 0.0,
      payment_type: 0.0,
      zip_count_4w: 0.0,
      velocity_6h: 0.0,
      velocity_24h,
      velocity_4w: 0.0,
      bank_branch_count_8w: 0.0,
      date_of_birth_distinct_emails_4w: 0.0,
      employment_status: 0.0,
      credit_risk_score,
      email_is_free: 0.0,
      housing_status: 0.0,
      phone_home_valid: 0.0,
      phone_mobile_valid: 0.0,
      bank_months_count: 0.0,
      has_other_cards: 0.0,
      proposed_credit_limit: 0.0,
      foreign_request: 0.0,
      source: 0.0,
      session_length_in_minutes: 0.0,
      device_os: 0.0,
      keep_alive_session: 0.0,
      device_distinct_emails_8w: 0.0,
      device_fraud_count,
      month: 0.0,
      x1: 0.0,
      x2: 0.0
    };

    try {
      const headers = { "Content-Type": "application/json" };
      if (API_KEY) headers["x-api-key"] = API_KEY;

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      rawDiv.textContent = JSON.stringify(data, null, 2);

      if (!res.ok) {
        resultDiv.style.display = "block";
        resultDiv.className = "result alert";
        resultDiv.textContent = "Error: " + (data.detail || "Prediction failed");
        return;
      }

      // Read fields from backend
      const prob = data.fraud_probability;
      const risk = data.risk_level;
      const isFraud = data.is_fraud;
      const alertTriggered = data.alert_triggered;
      const alertMessage = data.alert_message;

      resultDiv.style.display = "block";

      if (alertTriggered) {
        // High fraud risk / CRITICAL
        resultDiv.className = "result alert";
        resultDiv.textContent =
          (alertMessage || "⚠️ High-risk transaction detected") +
          `\nRisk level: ${risk}` +
          `\nFraud probability: ${(prob * 100).toFixed(2)}%`;
      } else {
        // Normal / low risk
        resultDiv.className = "result ok";
        resultDiv.textContent =
          `✅ Transaction considered safe\n` +
          `Risk level: ${risk}\n` +
          `Fraud probability: ${(prob * 100).toFixed(2)}%`;
      }

    } catch (err) {
      console.error(err);
      resultDiv.style.display = "block";
      resultDiv.className = "result alert";
      resultDiv.textContent = "Error calling API: " + err;
    }
  });
});
