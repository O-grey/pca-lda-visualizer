import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA / LDA Visualizer", layout="wide")
st.title("🔍 PCA and LDA Analyzer with Visualizations")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Dataset Preview", df.head())

    # Select class/label column
    st.markdown("#### 🎯 Select the class/label column (optional for PCA, required for LDA):")
    class_column = st.selectbox("Choose the class/label column", options=["None"] + list(df.columns))

    y = df[class_column] if class_column != "None" else None
    n_classes = len(y.unique()) if y is not None else None

    if y is not None:
        st.info(f"🔢 Number of unique classes in '{class_column}': {n_classes}")

    # Select analysis type
    analysis_type = st.selectbox("Choose analysis type", ["Select...", "PCA", "LDA"])

    # Filter only numeric features
    numeric_df = df.select_dtypes(include=['number'])

    # PCA Section
    if analysis_type == "PCA":
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numerical columns for PCA.")
        else:
            default_components = min(n_classes, len(numeric_df.columns)) if y is not None else 2
            n_components = st.slider(
                "Select number of components for PCA",
                1, min(len(numeric_df.columns), 10),
                default_components
            )

            scaled_data = StandardScaler().fit_transform(numeric_df)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)

            result_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
            if y is not None:
                result_df[class_column] = y

            st.write("### 🧪 PCA Result", result_df.head())
            st.download_button("📥 Download PCA Result CSV", result_df.to_csv(index=False), "pca_result.csv")

            # PCA 2D Scatter
            if n_components >= 2:
                fig = px.scatter(
                    result_df,
                    x="PC1", y="PC2",
                    color=class_column if class_column != "None" else None,
                    title="🔎 PCA 2D Projection",
                    width=800, height=500
                )
                st.plotly_chart(fig)

            # PCA Explained Variance
            st.write("### 📈 Explained Variance by Component")
            exp_var = pca.explained_variance_ratio_
            var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(exp_var))],
                "Explained Variance": exp_var
            })
            fig2 = px.bar(var_df, x="Component", y="Explained Variance", text_auto=True)
            st.plotly_chart(fig2)

    # LDA Section
    elif analysis_type == "LDA":
        if class_column == "None":
            st.warning("LDA requires a class/label column. Please select one.")
        else:
            features_df = df.drop(columns=[class_column])
            numeric_features = features_df.select_dtypes(include=['number'])

            if numeric_features.shape[1] < 1:
                st.warning("LDA requires at least one numeric feature.")
            else:
                scaled_features = StandardScaler().fit_transform(numeric_features)
                max_components = min(n_classes - 1, numeric_features.shape[1])

                if max_components < 1:
                    st.warning("LDA needs at least 2 classes and enough numeric features.")
                else:
                    if max_components == 1:
                        n_components = 1
                        st.info("Only 1 component possible for LDA with the current data.")
                    else:
                        n_components = st.slider("Select number of components for LDA", 1, max_components, 1)

                    lda = LDA(n_components=n_components)
                    lda_result = lda.fit_transform(scaled_features, y)

                    result_df = pd.DataFrame(lda_result, columns=[f"LD{i+1}" for i in range(n_components)])
                    result_df[class_column] = y
                    st.write("### 🧪 LDA Result", result_df.head())
                    st.download_button("📥 Download LDA Result CSV", result_df.to_csv(index=False), "lda_result.csv")

                    # LDA Visualization
                    if n_components == 1:
                        st.write("### 🎯 LDA 1D Visualization (Strip Plot)")
                        try:
                            fig = px.strip(
                                result_df,
                                x="LD1",
                                color=class_column,
                                title="🔎 LDA 1D Strip Plot",
                                stripmode="overlay",
                                width=800, height=300
                            )
                            fig.update_traces(jitter=0.3, marker=dict(size=8))
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"⚠️ Error generating LDA 1D plot: {e}")

                    elif n_components >= 2:
                        st.write("### 🎯 LDA 2D Visualization (Scatter Plot)")
                        try:
                            fig = px.scatter(
                                result_df,
                                x="LD1", y="LD2",
                                color=class_column,
                                title="🔎 LDA 2D Projection",
                                width=800, height=500
                            )
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"⚠️ Error generating LDA 2D plot: {e}")
