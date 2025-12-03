# app_cid_persist_shap_explain.py
from azure_helper import read_csv_from_blob, read_model_from_blob, read_json_from_blob
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from io import BytesIO
from typing import Optional, List, Tuple
import os
# SHAP + plotting
import shap
import matplotlib.pyplot as plt



st.set_page_config(page_title="Churn Prediction (by Customer ID)", layout="centered")
# NOTE: Do NOT call st.set_option("deprecation.showPyplotGlobalUse", False) – not supported on all versions.

# -----------------------------------------
# REQUIRED: function to be called before SHAP plot
# Replace the body with your actual OpenAI/LLM integration as needed.
# Arguments (as requested):
#   - customer_data: dict of raw input features for the selected row (data used to predict)
#   - shap_values:   list/array of SHAP values for the selected row (aligned to prediction_details['feature_names'])
#   - prediction_details: dict with keys:
#         'customer_id', 'prediction', 'churn_probability',
#         'threshold_used', 'base_value', 'feature_names'
# Returns: str (explanation to display above the SHAP graph)
# -----------------------------------------

from dotenv import load_dotenv

import os
import numpy as np
from openai import AzureOpenAI

load_dotenv()


# -----------------------------------------
# Constants
# -----------------------------------------
DEFAULT_INPUT_FILE = os.getenv("DEFAULT_INPUT_FILE_NAME")
CUSTOMER_ID_COL = os.getenv("CUSTOMER_ID_COL")
MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH")
MODEL_CN = os.getenv("MODEL_CN")
THRESHOLD_FN = os.getenv("THREASHOLD_FN")


# Azure OpenAI setup
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME", "gtp-4_try")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
connection_string = os.getenv("AZURE_BLOB_CS")
container_name = os.getenv("AZURE_BLOB_CN", "exports")
os.environ["AZURE_STORAGE_DISABLE_CERT_VALIDATION"] = "1"


client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

def openai_explain(customer_data, shap_values, prediction_details) -> str:
    feat_names = prediction_details.get("feature_names", [])
    base = prediction_details.get("base_value", None)
    prob = prediction_details.get("churn_probability", None)
    pred_label = "Churn" if int(prediction_details.get("prediction", 0)) == 1 else "Not Churn"

    sv = np.asarray(shap_values, dtype=float).flatten()
    k = min(3, len(sv))

    pairs = []
    for i, v in enumerate(sv):
        name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
        pairs.append((name, float(v)))

    pos = sorted([(n, v) for n, v in pairs if v > 0], key=lambda x: -abs(x[1]))[:k]
    neg = sorted([(n, v) for n, v in pairs if v < 0], key=lambda x: -abs(x[1]))[:k]

    # Construct prompt for Azure OpenAI
    prompt_lines = [
        f"The model predicted: **{pred_label}**",
        f"Churn probability: {prob:.2f}" if prob is not None else "",
        f"Base value: {base:.3f}" if base is not None else "",
        f"SHAP values summary:"
    ]
    for name, value in pairs:
        val = customer_data.get(name, "N/A")
        prompt_lines.append(f"- {name} (value: {val}): SHAP = {value:+.3f}")

    prompt_lines.append("Explain the prediction in simple terms, highlighting the most influential factors.")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that explains machine learning predictions in simple, human-readable language."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\n".join([line for line in prompt_lines if line])
                }
            ]
        }
    ]

    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        # max_tokens=1600,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    return completion.choices[0].message.content



# ============================================================
# Model wrapper: load pipeline + threshold and predict
# ============================================================
class XGBoostChurnPredictor:
    def __init__(self):
        self.pipeline = None
        self.threshold: float = 0.5
        self.threshold_strategy: Optional[str] = None
        self.expected_columns_: Optional[List[str]] = None

    def load_model(self, model_path: str) -> bool:
        """
        Loads:
        - .pkl bundle: {'pipeline': fitted pipeline, 'threshold': float, 'threshold_strategy': str}
        - (fallback) .joblib + threshold.json next to it
        """
        try:
            if model_path.endswith(".pkl"):
                # with open(model_path, "rb") as f:
                #     bundle = pickle.load(f)
                f = read_model_from_blob(connection_string, MODEL_CN, MODEL_BUNDLE_PATH)
                data = BytesIO(f)
                bundle = pickle.load(data)
                self.pipeline = bundle["pipeline"]
                self.threshold = float(bundle.get("threshold", 0.5))
                self.threshold_strategy = bundle.get("threshold_strategy")
            else:
                # Fallback: .joblib + threshold.json
                f = read_model_from_blob(connection_string, MODEL_CN, MODEL_BUNDLE_PATH)
                data = BytesIO(f)
                self.pipeline = joblib.load(data)
                thr_obj = read_json_from_blob(connection_string, MODEL_CN, THRESHOLD_FN)
                # thr_json = os.path.join(os.path.dirname(model_path), "threshold.json")
                try:
                    # with open(thr_json, "r") as f:
                    #     thr_obj = json.load(f)
                    self.threshold = float(thr_obj.get("threshold", 0.5))
                    self.threshold_strategy = thr_obj.get("strategy")
                except Exception:
                    self.threshold = 0.5
                    self.threshold_strategy = None

            self.expected_columns_ = self._extract_expected_columns()
            return True
        except Exception as e:
            st.error(f"Error loading model from '{model_path}': {e}")
            return False

    def _extract_expected_columns(self) -> Optional[List[str]]:
        """Derive the ORIGINAL input columns expected by the fitted ColumnTransformer."""
        try:
            prep = self.pipeline.named_steps.get("prep")
            if prep is None:
                return None
            expected = []
            for name, transformer, cols in prep.transformers_:
                if isinstance(cols, list):
                    expected.extend(cols)
                elif hasattr(cols, "__iter__"):
                    expected.extend(list(cols))
            return expected
        except Exception:
            return None

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align incoming df to training schema (add missing as NaN, drop extras, keep order)."""
        if not self.expected_columns_:
            return df.copy()
        out = df.copy()
        missing = [c for c in self.expected_columns_ if c not in out.columns]
        for c in missing:
            out[c] = np.nan
        out = out[self.expected_columns_]
        return out

    # ---- helpers to access prep/model and create transformed matrix with feature names
    def get_prep(self):
        try:
            return self.pipeline.named_steps.get("prep")
        except Exception:
            return None

    def get_estimator(self):
        # Try named 'model' first, else last step of pipeline
        try:
            if hasattr(self.pipeline, "named_steps"):
                steps = self.pipeline.named_steps
                if "model" in steps:
                    return steps["model"]
                return list(steps.values())[-1]
            return None
        except Exception:
            return None

    def transform_with_feature_names(self, X_in: pd.DataFrame):
        """
        Align -> transform using 'prep' -> return (np.array, feature_names)
        Works even if there's no 'prep' step.
        """
        X = self._align_columns(X_in)
        prep = self.get_prep()
        if prep is None:
            Xt = X
            feature_names = list(X.columns)
        else:
            Xt = prep.transform(X)
            try:
                feature_names = list(prep.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(Xt.shape[1])]

        # Convert to dense if needed without SciPy
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        else:
            Xt = np.asarray(Xt)
        return Xt, feature_names

    def predict_subset(
        self,
        df_all: pd.DataFrame,
        id_col: str,
        id_values: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filters df_all by id_col in id_values, aligns columns, predicts, returns:
        - preds_df (ID, probability, prediction, label, threshold_used)
        - missing_ids (list of IDs that didn't match any row)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded.")

        df_all = df_all.copy()
        if id_col not in df_all.columns:
            raise ValueError(f"ID column '{id_col}' not found in input data.")
        df_all[id_col] = df_all[id_col].astype(str)
        id_values_norm = [str(v).strip() for v in id_values if str(v).strip() != ""]
        id_set = set(id_values_norm)

        # Filter
        mask = df_all[id_col].isin(id_set)
        sub = df_all.loc[mask].copy()
        missing_ids = sorted(list(id_set - set(sub[id_col].unique())))
        if sub.empty:
            return pd.DataFrame(), missing_ids

        # Drop any target-like columns if present
        for tgt in ["churn", "Churn", "target"]:
            if tgt in sub.columns:
                sub = sub.drop(columns=[tgt])

        X = self._align_columns(sub)

        # Predict
        probs = self.pipeline.predict_proba(X)[:, 1]
        thr = float(self.threshold)
        preds = (probs >= thr).astype(int)
        labels = np.where(preds == 1, "Yes", "No")

        out = pd.DataFrame({
            id_col: sub[id_col].values,
            "churn_probability": probs,
            "prediction": preds,
            "Churn": labels,
            "threshold_used": thr,
        })

        # Keep same order as provided IDs
        order_map = {cid: i for i, cid in enumerate(id_values_norm)}
        out["_order"] = out[id_col].map(order_map)
        out = out.sort_values(["_order", id_col]).drop(columns=["_order"])
        return out, missing_ids


# -----------------------------------------
# Cache: model and background data
# -----------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_cached(model_path: str):
    predictor = XGBoostChurnPredictor()
    ok = predictor.load_model(model_path)
    if ok:
        return predictor
    return None

@st.cache_resource(show_spinner=True)
def load_input_data_bg(csv_path: str) -> pd.DataFrame:
    try:
        # df = pd.read_csv(csv_path)
        df = read_csv_from_blob(connection_string, container_name , csv_path)
        # print("df from blob "+df.to_csv())
        return df
    except FileNotFoundError:
        st.error(f"Input file '{csv_path}' not found. Place it alongside this app or update the path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read input file '{csv_path}': {e}")
        return pd.DataFrame()

# ===========================
# UI
# ===========================
st.title("Customer Churn Prediction")

# Load model and background data
predictor = load_model_cached(MODEL_BUNDLE_PATH)
if predictor is None:
    st.stop()
df_bg = load_input_data_bg(DEFAULT_INPUT_FILE)
if df_bg.empty:
    st.stop()

# Strict requirement: the file must contain the customer_id column
if CUSTOMER_ID_COL not in df_bg.columns:
    st.error(f"The input file does not contain the required ID column '{CUSTOMER_ID_COL}'.")
    st.stop()

# --- Session-state buckets (persist across reruns)
ss = st.session_state
for k in [
    "preds_df",
    "missing_ids",
    "explained_ids",
    "Xt",
    "feat_names",
    "shap_values",           # normalized to class-1 contributions
    "expected_base_value",   # scalar base value aligned to shap_values
    "selected_cid",
    "raw_rows",              # <--- NEW: raw input rows aligned to preds_df order (data used to predict)
]:
    ss.setdefault(k, None)

# ----- INPUT UX: Suggestions + Free text
st.markdown("Enter one or more **customer_id** values (use type-ahead picker or paste comma/newline separated):")
c1, c2 = st.columns([1, 1], vertical_alignment="top")

with c1:
    all_ids = sorted(df_bg[CUSTOMER_ID_COL].astype(str).unique().tolist())
    picked_ids = st.multiselect(
        "Pick Customer ID(s)",
        options=all_ids,
        placeholder="Type to search & select…",
        help="Start typing to search (supports multiple selection)."
    )

with c2:
    id_text = st.text_area(
        "Or paste Customer ID(s)",
        placeholder="e.g.\nC001\nC102\nC205",
        height=120
    )

# Parse IDs from text area
typed_ids = []
if id_text.strip():
    raw = id_text.replace(",", "\n").split("\n")
    typed_ids = [r.strip() for r in raw if r.strip()]

# Final ID list (preserve order: picked first, then typed)
final_ids = list(dict.fromkeys(list(picked_ids) + typed_ids))

# Predict button: compute predictions & SHAP once, store in session_state
if st.button("Predict"):
    if not final_ids:
        st.error("Please provide at least one customer_id.")
    else:
        try:
            preds_df, missing_ids = predictor.predict_subset(
                df_all=df_bg,
                id_col=CUSTOMER_ID_COL,
                id_values=final_ids
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            preds_df, missing_ids = pd.DataFrame(), []

        if preds_df.empty:
            st.error("No matching customer_id found in the background file.")
            # Clear any prior results so UI doesn't show stale content
            ss.preds_df = None
        else:
            # Keep predictions and warnings
            ss.preds_df = preds_df
            ss.missing_ids = missing_ids

            # --- Build SHAP-ready subset in the same order as preds_df
            df_bg_tmp = df_bg.copy()
            df_bg_tmp[CUSTOMER_ID_COL] = df_bg_tmp[CUSTOMER_ID_COL].astype(str)

            explained_ids = preds_df[CUSTOMER_ID_COL].astype(str).tolist()
            found_set = set(explained_ids)
            sub = df_bg_tmp[df_bg_tmp[CUSTOMER_ID_COL].isin(found_set)].copy()

            # Drop any target-like columns if present
            for tgt in ["churn", "Churn", "target"]:
                if tgt in sub.columns:
                    sub = sub.drop(columns=[tgt])

            # Order to match preds_df
            order_map = {cid: i for i, cid in enumerate(explained_ids)}
            sub["_order"] = sub[CUSTOMER_ID_COL].map(order_map)
            sub = sub.sort_values(["_order", CUSTOMER_ID_COL]).drop(columns=["_order"]).reset_index(drop=True)

            # Save raw_rows (data used to predict) for later openai_explain()
            ss.raw_rows = sub.copy()

            # Transform once for SHAP + feature names
            Xt, feat_names = predictor.transform_with_feature_names(sub)

            # Compute SHAP once for the whole predicted subset
            est = predictor.get_estimator()
            shap_values = None
            expected_value = None
            if est is not None:
                with st.spinner("Computing SHAP values…"):
                    try:
                        explainer = shap.TreeExplainer(est)
                        shap_values = explainer.shap_values(Xt)
                        expected_value = explainer.expected_value
                    except Exception:
                        # Fallback (model-agnostic)
                        explainer = shap.Explainer(est, Xt)
                        sv_exp = explainer(Xt)
                        shap_values = sv_exp.values
                        expected_value = sv_exp.base_values

            # Normalize SHAP return types to 2D array for class 1 + scalar base value
            if shap_values is not None:
                if isinstance(shap_values, list):
                    sv_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    if isinstance(expected_value, (list, np.ndarray)):
                        base = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else np.atleast_1d(expected_value)[0]
                    else:
                        base = expected_value
                else:
                    sv_arr = shap_values
                    if isinstance(expected_value, (list, np.ndarray)):
                        base = np.atleast_1d(expected_value)[0]
                    else:
                        base = expected_value
            else:
                sv_arr, base = None, None

            # Save into session state so UI persists across reruns
            ss.explained_ids = explained_ids
            ss.Xt = Xt
            ss.feat_names = feat_names
            ss.shap_values = sv_arr
            ss.expected_base_value = base

            # Default selected customer (preserve prior choice if still valid)
            if ss.selected_cid not in set(explained_ids):
                ss.selected_cid = explained_ids[0] if explained_ids else None

# ===== Persisted RESULTS & EXPLANATION (shown on every rerun if available) =====
if ss.preds_df is not None:
    preds_df = ss.preds_df

    # Show missing id warnings (if any)
    if ss.missing_ids:
        show_list = ", ".join(ss.missing_ids[:25])
        more = f" (+{len(ss.missing_ids)-25} more)" if len(ss.missing_ids) > 25 else ""
        st.error(f"The following customer_id(s) were not found: {show_list}{more}")

    st.success(f"✅ Predictions generated for {len(preds_df)} customer(s).")
    st.dataframe(preds_df, use_container_width=True)

    # Downloads
    st.download_button(
        "Download CSV",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions_by_id.csv",
        mime="text/csv",
        use_container_width=True
    )

    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        preds_df.to_excel(writer, index=False, sheet_name="predictions")
    st.download_button(
        "Download Excel",
        data=excel_buf.getvalue(),
        file_name="churn_predictions_by_id.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # -------- SHAP explanation section (persists across selection changes)
    with st.expander("🔍 Explain a prediction (SHAP)", expanded=True):
        if not ss.explained_ids or ss.Xt is None or ss.shap_values is None:
            st.info("SHAP explanation not available for these predictions.")
        else:
            # Keep the last selected customer across reruns
            try:
                default_index = ss.explained_ids.index(ss.selected_cid) if ss.selected_cid else 0
            except ValueError:
                default_index = 0

            sel_cid = st.selectbox(
                "Select a Customer ID to explain",
                options=ss.explained_ids,
                index=default_index,
                key="selected_cid"   # persists automatically
            )


            if st.button("Explain"):
            # if sel_cid:
                row_idx = ss.explained_ids.index(sel_cid)

                # Build Explanation object for the selected row
                try:
                    ex = shap.Explanation(
                        values=ss.shap_values[row_idx],
                        base_values=ss.expected_base_value,
                        data=ss.Xt[row_idx],
                        feature_names=ss.feat_names
                    )
                except Exception:
                    ex = shap.Explanation(
                        values=ss.shap_values[row_idx],
                        base_values=ss.expected_base_value
                    )

                # ------- NEW: call openai_explain BEFORE the plot
                # Prepare inputs for the function
                # customer_data used to predict: raw row aligned to preds_df order
                customer_raw_row = {}
                if ss.raw_rows is not None and 0 <= row_idx < len(ss.raw_rows):
                    # Convert to dict of raw features used to predict
                    customer_raw_row = ss.raw_rows.iloc[row_idx].to_dict()

                # prediction details for the selected row
                row_pred = preds_df[preds_df[CUSTOMER_ID_COL] == sel_cid].iloc[0]
                prediction_details = {
                    "customer_id": sel_cid,
                    "prediction": int(row_pred["prediction"]),
                    "churn_probability": float(row_pred["churn_probability"]),
                    "threshold_used": float(row_pred["threshold_used"]),
                    "base_value": float(ss.expected_base_value) if ss.expected_base_value is not None else None,
                    "feature_names": ss.feat_names,
                }

                try:
                    summary_text = openai_explain(
                        customer_data=customer_raw_row,
                        shap_values=ss.shap_values[row_idx],
                        prediction_details=prediction_details
                    )
                except Exception as e:
                    summary_text = f"(Explanation function failed: {e})"

                # ---- Display the model-generated explanation string BEFORE the SHAP graph
                st.markdown("**Explanation**")
                st.write(summary_text)

                # ---- Then show the usual header + quick facts + SHAP plot
                st.subheader("SHAP values plot")
                st.write(
                    f"**Customer:** `{sel_cid}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Prediction:** {'Churn' if int(row_pred['prediction'])==1 else 'Not Churn'} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Churn Probability:** {float(row_pred['churn_probability']):.2f}"
                )

                fig, _ = plt.subplots(figsize=(9, 5))
                shap.plots.waterfall(ex, max_display=10, show=False)
                st.pyplot(fig, use_container_width=True)

# (Optional) Model details
with st.expander("Model details", expanded=False):
    st.write(
        f"Stored decision threshold: **{getattr(predictor, 'threshold', 0.5):.3f}**"
        + (f" _(strategy: {predictor.threshold_strategy})_" if predictor.threshold_strategy else "")
    )
    st.caption(f"Input file: `{DEFAULT_INPUT_FILE}`\nID column: `{CUSTOMER_ID_COL}`")


