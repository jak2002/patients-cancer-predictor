import matplotlib.pyplot as plt
import io, base64
import numpy as np

def get_feature_importance_plot(model_rf, model_svm, feature_names):
    """
    Generate side-by-side feature importance plots for Random Forest and SVM.
    Returns a base64-encoded PNG image string.
    """

    # Random Forest importances
    rf_importances = model_rf.feature_importances_

    # SVM coefficients (absolute values for importance)
    if hasattr(model_svm, "coef_"):
        svm_importances = np.abs(model_svm.coef_[0])
    else:
        svm_importances = np.zeros(len(feature_names))  # fallback if not available

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Random Forest plot
    axes[0].barh(feature_names, rf_importances, color="skyblue")
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].set_xlabel("Importance")

    # SVM plot
    axes[1].barh(feature_names, svm_importances, color="lightcoral")
    axes[1].set_title("SVM Feature Importance (coefficients)")
    axes[1].set_xlabel("Magnitude")

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return img_base64
