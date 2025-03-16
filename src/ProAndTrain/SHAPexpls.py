import shap as sh

#explainer = sh.Explainer(model.get_model(), X_train)
#shap_values = explainer(X_test)

def GetSummaryPLot(dataFrame, X_test, shap_values):
    return sh.summary_plot(shap_values, X_test, feature_names=dataFrame.drop(columns=['Biopsy']).columns)

def GetDependencePlot(dataFrame, X_test, shap_values):
    return sh.dependence_plot('age', shap_values, X_test, feature_names=dataFrame.drop(columns=['Biopsy']).columns)

def GetDependencePlot(dataFrame, X_test, shap_values):
    return sh.force_plot(shap_values[0].base_values, shap_values[0].values, X_test[0], feature_names=dataFrame.drop(columns=['Biopsy']).columns)