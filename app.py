import os
import joblib
import numpy as np
import gradio as gr

# Load the trained model
model = joblib.load("loan_approval_model.pkl")


def predict_loan(
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):
    try:
        features = np.array([[
            float(income_annum),
            float(loan_amount),
            int(loan_term),
            int(cibil_score),
            float(residential_assets_value),
            float(commercial_assets_value),
            float(luxury_assets_value),
            float(bank_asset_value)
        ]])

        prediction = model.predict(features)

        # If model predicts 0/1
        if prediction[0] == 1:
            return "Loan Approved"
        else:
            return "Loan Rejected"

        # If  model predicts strings instead, replace the above with:
        # if prediction[0] == "Approved":
        #     return " Loan Approved"
        # else:
        #     return " Loan Rejected"

    except Exception as e:
        return f"Error: {e}"


demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="No of Dependents"),
        gr.Number(label="Education"),
        gr.Number(label="Self Employed"),
        gr.Number(label="Income Per Annum"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term (Months)"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Loan Approval Prediction System",
    description="""
<h3>Project by Parth - PIET (241504)</h3>
<p>Enter the applicant's financial details to predict whether the loan will be approved.</p>
""",
    theme=gr.themes.Soft()
)

import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
