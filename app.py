# 1. Imports and page setup
 
# Streamlit for the dashboard interface, with pandas and numpy for the data
import streamlit as st
import pandas as pd
import numpy as np
 
# The fairness metrics and the error rates from Fairlearn for comparing the two groups
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               demographic_parity_ratio, equalized_odds_difference,
                               false_negative_rate, false_positive_rate)
# Accuracy, precision and recall, and the ROC curve and its area, for measuring and comparing performance by sex
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
 
# Matplotlib for the panels that draw the SHAP importances and the calibration curves
from matplotlib import pyplot as plt
 
# SHAP for rebuilding the local waterfall explanation of a single patient
import shap
 
# Use the full width of the browser, since the dashboard has several panels side by side
st.set_page_config(page_title='Fairness Trade-off Dashboard', layout='wide')
 
# Theme colours and background are set in .streamlit/config.toml; this block only adds the fine details config.toml cannot reach, such as the metric sizing, the card shadows, the heading styling and the tab styling
st.markdown('''
    <style>
    /* Shrink the large metric numbers to roughly half their default size */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    /* Keep the metric label readable alongside the smaller value */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    /* Set the caption text to a clear, readable size for the panel explanations */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        font-size: 0.95rem;
    }
    /* Give each metric a white card look with a soft border and shadow, so the boxes lift off the grey background */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 3px rgba(31, 59, 111, 0.06);
    }
    /* Style the page title in navy with a thin underline, so it reads as a clear branded header */
    h1 {
        color: #1f3b6f !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }
    /* Mark each panel heading in navy with a short left bar, with a balanced gap to the text and room below so the caption underneath is not cramped */
    h2, h3 {
        color: #1f3b6f !important;
        border-left: 4px solid #1f3b6f;
        padding-left: 0.5rem !important;
        margin-left: 0;
        margin-bottom: 0.6rem !important;
    }
    /* Sit the six tabs in a light grey strip with spacing between them, so it is clear they are separate sections that can be switched */
    .stTabs [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        background-color: #eef2f7;
        padding: 0.4rem;
        border-radius: 0.6rem;
    }
    /* Give each tab a white button look with a border, so the six sections are easy to tell apart and read as clickable */
    .stTabs [data-baseweb="tab"],
    [data-testid="stTab"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    /* Set the tab label text size on the inner text element, since the label sits inside a paragraph within the tab */
    .stTabs [data-baseweb="tab"] p,
    [data-testid="stTab"] p {
        font-size: 0.95rem;
    }
    /* Fill the selected tab in navy, so the current section stands out clearly from the others */
    .stTabs [aria-selected="true"],
    [data-testid="stTab"][aria-selected="true"] {
        background-color: #1f3b6f;
        border-color: #1f3b6f;
    }
    /* Turn the selected tab label white, since its text sits in an inner paragraph */
    .stTabs [aria-selected="true"] p,
    [data-testid="stTab"][aria-selected="true"] p {
        color: #ffffff;
    }
    /* Hide the selection indicator line, since the navy fill already marks the current tab */
    .stTabs [data-baseweb="tab-highlight"],
    [data-testid="stTab"] .react-aria-SelectionIndicator,
    [data-testid="stTab"] [class*="SelectionIndicator"] {
        display: none;
    }
    /* Keep the control panel text at a steady size, so the controls read the same across Streamlit versions rather than shrinking on the deployed one */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        font-size: 0.90rem;
    }
    </style>
    ''', unsafe_allow_html=True)
 
 
# 2. Datasets, labels and data loading
 
# Each dataset has its own set of saved files, and the group column is named sex or gender
DATASETS = {'UCI Heart Disease': {'prefix': 'uci', 'group': 'sex'},
            'Kaggle Cardiovascular Disease': {'prefix': 'kaggle', 'group': 'gender'}}
 
# The readable name for each feature, so the charts show clear labels instead of the raw column names
READABLE_LABELS = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain', 'trestbps': 'Blood Pressure',
                   'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
                   'thalch': 'Max Heart Rate', 'exang': 'Exercise Angina', 'oldpeak': 'ST Depression',
                   'gender': 'Gender', 'height': 'Height', 'weight': 'Weight', 'ap_hi': 'Systolic BP',
                   'ap_lo': 'Diastolic BP', 'cholesterol': 'Cholesterol', 'gluc': 'Glucose',
                   'smoke': 'Smoking', 'alco': 'Alcohol', 'active': 'Active'}
 
# The three sets of probabilities the user can switch between, so the dashboard can show the baseline and the two mitigations
MITIGATIONS = {'Baseline': 'probability', 'SMOTE-NC': 'probability_smote', 'Reweighting': 'probability_reweight'}
 
# The file suffix for each mitigation method, so the matching calibration files can be found by name
CALIBRATION_SUFFIX = {'Baseline': 'baseline', 'SMOTE-NC': 'smote', 'Reweighting': 'reweight'}
 
# The plain-language meaning of each fairness metric, shown as an (i) tooltip next to it on the fairness metrics panel
METRIC_HELP = {
    'Demographic Parity Difference': 'Whether the model flags disease at the same rate for both sexes. It measures the '
                                     'gap between the share of female patients and the share of male patients predicted '
                                     'to have disease, so a value of 0 means both groups are flagged equally often.',
    'Equalised Odds Difference': 'Whether the model is equally accurate for both sexes, looking separately at the '
                                 'patients who truly have disease and those who do not. It brings together the gap in '
                                 'the true positive rate and the gap in the false positive rate, so a value of 0 means '
                                 'the model makes the same kinds of error for each group.',
    'Predictive Parity Difference': 'Whether a positive prediction can be trusted equally for both sexes. It is the gap '
                                    'in precision between the groups, precision being the share of flagged patients who '
                                    'truly have disease, so a value of 0 means a \'disease\' flag carries the same '
                                    'weight for a female and a male patient.',
    'Disparate Impact Ratio': 'The ratio of the two groups\' positive-prediction rates. A ratio of 1 means both groups '
                              'are flagged equally often. In this dashboard, the ratio is judged against the tolerance '
                              'set in the control panel, so it responds to the same fairness bar as the other '
                              'threshold-dependent metrics. The 0.8 line (the four-fifths rule) is also shown in the '
                              'comparison view as a common reference point.'}
 
# The plain-language meaning of each performance measure, shown as an (i) tooltip next to it on the performance panel
PERF_HELP = {
    'Recall': 'The share of patients who truly have disease that the model correctly flags, so a higher value is '
              'better here. It is also known as the true positive rate, or sensitivity.',
    'Precision': 'Among the patients flagged as having disease, the share who truly have it, so a higher value is the '
                 'better outcome.',
    'False Negative Rate': 'The share of patients who truly have disease but are predicted healthy. These are missed '
                           'cases, and since a missed case is usually more serious than a false alarm in a clinical '
                           'setting, a lower value is preferable.',
    'False Positive Rate': 'The share of healthy patients wrongly flagged as having disease. These are false alarms, '
                           'which lead to unnecessary tests, so a lower value is preferable here.'}
 
# The description of every feature the model uses, shown as a table on the overview tab so each column can be understood
FEATURE_INFO = {
    'uci': [('Age', 'Age of the patient in years'),
            ('Sex', 'Sex of the patient, male or female'),
            ('Chest Pain', 'Chest pain type: typical angina, atypical angina, non-anginal or asymptomatic'),
            ('Blood Pressure', 'Resting blood pressure on admission, in mm Hg'),
            ('Cholesterol', 'Serum cholesterol in mg/dl'),
            ('Fasting Blood Sugar', 'Whether fasting blood sugar is greater than 120 mg/dl'),
            ('Resting ECG', 'Resting electrocardiographic result: normal, ST-T abnormality or LV hypertrophy'),
            ('Max Heart Rate', 'Maximum heart rate achieved'),
            ('Exercise Angina', 'Whether exercise induced angina, true or false'),
            ('ST Depression', 'ST depression induced by exercise relative to rest')],
    'kaggle': [('Age', 'Age of the patient, converted from days to years'),
               ('Gender', 'Gender of the patient, women or men'),
               ('Height', 'Height of the patient in centimetres'),
               ('Weight', 'Weight of the patient in kilograms'),
               ('Systolic BP', 'Systolic blood pressure'),
               ('Diastolic BP', 'Diastolic blood pressure'),
               ('Cholesterol', 'Cholesterol level: normal, above normal or well above normal'),
               ('Glucose', 'Glucose level: normal, above normal or well above normal'),
               ('Smoking', 'Whether the patient smokes'),
               ('Alcohol', 'Whether the patient drinks alcohol'),
               ('Active', 'Whether the patient is physically active')]}
 
# A short note on how each dataset was prepared, so the overview tab can explain the patient counts
DATASET_NOTES = {
    'uci': 'The UCI data combines four hospitals. Some columns from the original data were dropped during '
           'preprocessing because too much of their data was missing to impute reliably.',
    'kaggle': 'The Kaggle data began with 70,000 records. Rows with biologically implausible values, treated as data '
              'entry errors, were removed during preprocessing, which reduced the patient count to the figure shown above.'}
 
# The full patient count after preprocessing for each dataset, kept here since the saved probabilities hold only the test set the dashboard works on
DATASET_TOTAL = {'uci': 920, 'kaggle': 68653}
 
# Load the probabilities and base rates for a dataset once, so the dashboard does not read them again on every rerun
@st.cache_data
def load_data(prefix):
    """Read the probabilities and the base rates that were saved from the notebook for one dataset."""
    probabilities = pd.read_csv('dashboard_data/{}_probabilities.csv'.format(prefix))
    base_rate = pd.read_csv('dashboard_data/{}_base_rate.csv'.format(prefix))
    return probabilities, base_rate
 
# Load the gender-stratified SHAP importances for a dataset once, like the other data
@st.cache_data
def load_shap_gender(prefix):
    """Read the mean absolute SHAP value of each feature for the female and male patients."""
    return pd.read_csv('dashboard_data/{}_shap_gender.csv'.format(prefix))
 
# Load the saved calibration curve points for a dataset and mitigation method, like the other data
@st.cache_data
def load_calibration(prefix, method):
    """Read the calibration curve points for the female and male patients under one mitigation method."""
    female = pd.read_csv('dashboard_data/{}_calibration_female_{}.csv'.format(prefix, method))
    male = pd.read_csv('dashboard_data/{}_calibration_male_{}.csv'.format(prefix, method))
    return female, male
 
# Load the per-patient SHAP values and scaled feature values for a dataset once, like the other data
@st.cache_data
def load_local(prefix):
    """Read the individual SHAP values, the scaled feature values and the base value for every patient."""
    shap_values = pd.read_csv('dashboard_data/{}_shap_values.csv'.format(prefix))
    feature_values = pd.read_csv('dashboard_data/{}_feature_values.csv'.format(prefix))
    base_value = pd.read_csv('dashboard_data/{}_base_value.csv'.format(prefix))['base_value'].values[0]
    return shap_values, feature_values, base_value
 
 
# 3. Metric calculation and local waterfall helper
 
# A precision that returns 0 instead of warning when a group has no positive predictions, since extreme thresholds can flag no one in a group
def safe_precision(y_true, y_pred):
    """Return precision, treating the no-positive-prediction case as 0 rather than raising a warning."""
    return precision_score(y_true, y_pred, zero_division=0)
 
# Measure the four fairness metrics from a set of predictions, so the same function works for a threshold and for the threshold optimiser
def compute_metrics(y_true, y_pred, group):
    """Take a set of predictions and return the four fairness metrics for the two groups."""
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=group)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=group)
    di = demographic_parity_ratio(y_true, y_pred, sensitive_features=group)
    precision_by_group = MetricFrame(metrics=safe_precision, y_true=y_true, y_pred=y_pred,
                                     sensitive_features=group)
    pp = precision_by_group.difference()
 
    return {'Demographic Parity Difference': dp, 'Equalised Odds Difference': eo,
            'Predictive Parity Difference': pp, 'Disparate Impact Ratio': di}
 
# Turn one method's probabilities into predictions at the current thresholds and return its four fairness metrics with accuracy and recall, so the comparison can be built live for a dataset
def metrics_for_method(probabilities, group_column, method, female_threshold, male_threshold):
    """Apply the current thresholds to one method for one dataset and return its four fairness metrics with accuracy and recall."""
    group = probabilities[group_column].values
    y_true = probabilities['y_true'].values
 
    # The threshold optimiser has a fixed decision, while the other methods turn a probability into a decision with the thresholds
    if method == 'Threshold Optimiser':
        predictions = probabilities['threshold_pred'].values
    else:
        proba = probabilities[MITIGATIONS[method]].values
        female_mask = group == 0
        male_mask = group == 1
        predictions = np.zeros(len(group), dtype=int)
        predictions[female_mask] = (proba[female_mask] >= female_threshold).astype(int)
        predictions[male_mask] = (proba[male_mask] >= male_threshold).astype(int)
 
    # The four fairness metrics come from the shared helper, and accuracy and recall are added for the overall picture
    scores = compute_metrics(y_true, predictions, group)
    scores['Accuracy'] = accuracy_score(y_true, predictions)
    scores['Recall'] = recall_score(y_true, predictions)
    return scores
 
# Rebuild the SHAP explanation for one patient and draw the waterfall, so the same plot can be reused for any selected patient
def draw_local_waterfall(patient, shap_values, base_value, feature_values):
    """Build the SHAP explanation for one patient and draw the waterfall plot, returning the figure."""
    explanation = shap.Explanation(values=shap_values.values[patient],
                                   base_values=base_value,
                                   data=feature_values.values[patient],
                                   feature_names=list(shap_values.columns.map(READABLE_LABELS)))
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    return fig
 
 
# 4. Sidebar controls and page layout
 
# Give the dashboard a title and a short plain-language description of what it does
st.title('Fairness Trade-off Dashboard')
st.write('This dashboard examines whether a tuned XGBoost heart disease model treats female and male patients fairly. By changing '
         'how the model makes its decision, you can watch the fairness measures respond as you go. The aim is not to '
         'arrive at a single fair model, but to bring a trade-off into view: when the two groups have different disease '
         'rates, satisfying every fairness measure at once becomes impossible or very difficult, and improving one '
         'measure tends to come at the cost of another. What this dashboard offers is a way to see where that trade-off falls.')
 
# List the three mitigation methods on their own, since the baseline is the untouched model rather than a mitigation
mitigation_methods = ['SMOTE-NC', 'Reweighting', 'Threshold Optimiser']
 
# Put all the controls in the control panel, so they stay in one place while the user moves between the tabs
with st.sidebar:
    st.header('Controls')
    dataset_name = st.selectbox('Dataset', list(DATASETS.keys()),
                                help='Choose which dataset to explore. Every panel updates to reflect the one you pick.')
 
    # Work out the prefix now, since the panels below load their data by it
    config = DATASETS[dataset_name]
    prefix = config['prefix']
 
    # Keep the baseline separate from the mitigations, since it is the untouched model that the mitigations are compared against
    use_mitigation = st.radio('Model', ['Baseline', 'Apply a mitigation'],
                              help='The baseline is the tuned XGBoost model before any fairness mitigation is applied. A mitigation is a method that tries to make it fairer.')
    if use_mitigation == 'Apply a mitigation':
        mitigation_name = st.radio('Mitigation Method', mitigation_methods,
                                   help='SMOTE-NC and reweighting alter the training data and retrain the model, which '
                                        'changes the probability scores. The threshold optimiser leaves the model '
                                        'untouched and simply picks a separate threshold for each sex, so the scores '
                                        'themselves stay the same.')
    else:
        mitigation_name = 'Baseline'
 
    # The threshold optimiser sets its own threshold for each sex, so the sliders are switched off when it is chosen
    optimiser_chosen = mitigation_name == 'Threshold Optimiser'
    female_threshold = st.slider('Female Threshold', 0.0, 1.0, 0.5, 0.01, disabled=optimiser_chosen,
                                 help='The threshold applied to female patients. If the model\'s score for a patient '
                                      'reaches this value or higher, she is flagged as having disease. Lowering the '
                                      'threshold flags more of them, raising it flags fewer.')
    male_threshold = st.slider('Male Threshold', 0.0, 1.0, 0.5, 0.01, disabled=optimiser_chosen,
                               help='The threshold applied to male patients. If the model\'s score for a patient reaches '
                                    'this value or higher, he is flagged as having disease. Setting the two thresholds '
                                    'apart from each other is one way to trade fairness against accuracy between the groups.')
    tolerance = st.slider('Tolerance', 0.0, 0.5, 0.1, 0.01,
                          help='How large a gap between the two groups still counts as fair. On the Fairness Metrics '
                               'tab a metric turns green when it sits within this tolerance and red when it falls '
                               'outside. Think of it as your own fairness bar: moving it shows which metrics pass under '
                               'a strict setting and which only pass under a looser one.')
 
    # Show a short status line so the user can see at a glance whether the thresholds are on the default, adjusted, or set by the optimiser
    if optimiser_chosen:
        st.info('The threshold optimiser sets its own threshold for each sex, so the sliders above are switched off.')
    elif female_threshold == 0.5 and male_threshold == 0.5:
        st.success('Both thresholds are on the default of 0.50.')
    else:
        st.warning('Thresholds adjusted to {:.2f} for female and {:.2f} for male patients, away from the 0.50 '
                   'default.'.format(female_threshold, male_threshold))
 
# Load the chosen dataset and pull out the columns the panels need
probabilities, base_rate = load_data(prefix)
group = probabilities[config['group']].values
y_true = probabilities['y_true'].values

# The word for the protected attribute differs by dataset, so it is set once here and reused in the headings and notes below
# UCI records it as sex, while the Kaggle data records it as gender, matching each notebook
group_label = 'Sex' if prefix == 'uci' else 'Gender'
 
# The threshold optimiser returns a fixed decision, while the other methods give a probability that the thresholds turn into a decision
# The baseline probability is still loaded for the panels that need a score, since the optimiser does not provide one
proba = probabilities['probability'].values
if optimiser_chosen:
    predictions = probabilities['threshold_pred'].values
else:
    proba = probabilities[MITIGATIONS[mitigation_name]].values
    female_mask = group == 0
    male_mask = group == 1
    predictions = np.zeros(len(group), dtype=int)
    predictions[female_mask] = (proba[female_mask] >= female_threshold).astype(int)
    predictions[male_mask] = (proba[male_mask] >= male_threshold).astype(int)
 
# Point users to the tab strip below, so they notice the dashboard has six sections to move through
st.write('**Use the six tabs below to move between sections:** Dataset Overview, Fairness Metrics, Explanation, Calibration & ROC, Errors, and Dataset and Metric Comparison.')
# Split the dashboard into six tabs, so each area of the analysis has its own place
tab_overview, tab_tradeoff, tab_explain, tab_quality, tab_errors, tab_compare = st.tabs(
    ['Dataset Overview', 'Fairness Metrics', 'Explanation', 'Calibration & ROC', 'Errors', 'Dataset and Metric Comparison'])
 
with tab_overview:

    # 5. Dataset overview

    # Give a plain description of the dataset in view, so the user knows what they are looking at before the analysis
    st.subheader('About This Dataset')
    st.caption('This tab describes the dataset currently selected in the control panel. It gives the size of the data and the '
               'meaning of every feature the model uses. Switching the dataset in the control panel updates everything shown '
               'here, so each dataset can be read on its own terms.')

    # Show the full size after preprocessing, the size of the test set the dashboard works on, and the number of features
    # The test-set count is read live from the loaded data, while the full count is held in a dict since the saved data holds only the test set
    n_test = len(probabilities)
    n_features = len(FEATURE_INFO[prefix])
    overview_total, overview_test, overview_features = st.columns(3)
    overview_total.metric('Total Patients', '{}'.format(DATASET_TOTAL[prefix]),
                          help='The number of patients left after cleaning the raw data, before the split into training and test sets.')
    overview_test.metric('Test Set (shown here)', '{}'.format(n_test),
                         help='The held-out portion the dashboard runs on. Every panel is worked out on these patients, so the fairness measures and error tables all refer to this set.')
    overview_features.metric('Features Used', '{}'.format(n_features))

    # Add the short preparation note, so the patient count and the choice of features make sense
    st.info(DATASET_NOTES[prefix])

    st.divider()

    # List every feature the model uses with a plain description, so the SHAP charts on the Explanation tab can be read with confidence
    st.subheader('Features Used by the Model')
    st.caption('These are the patient details the model draws on to make its prediction. The same names appear on the '
               'Explanation tab, so this table can be used as a reference for what each one means.')
    feature_table = pd.DataFrame(FEATURE_INFO[prefix], columns=['Feature', 'Description'])
    st.dataframe(feature_table, hide_index=True, use_container_width=True)


with tab_tradeoff:


    # 6. Disease rate in the data

    # Show the disease rate for each sex, which is the context that decides how sharp the trade-off can be
    st.subheader('Disease Rate by {} (Base Rate)'.format(group_label))
    st.caption('This shows the share of female and male patients who actually have heart disease in the data, a figure '
               'known as the base rate. When the gap between the two groups is large, keeping the model fair for both at '
               'once becomes mathematically harder, and that is the trade-off this dashboard sets out to explore. A '
               'large gap on its own is not proof of bias, though. It might reflect a genuine difference in how often '
               'the disease occurs, or it might reflect a history of under-diagnosis in one group, and the data alone '
               'cannot tell the two apart.')
    female_rate = base_rate['disease_rate'].values[0]
    male_rate = base_rate['disease_rate'].values[1]
    gap = male_rate - female_rate
    rate_female, rate_male, rate_gap = st.columns(3)
    rate_female.metric('Female', '{:.2f}%'.format(female_rate * 100))
    rate_male.metric('Male', '{:.2f}%'.format(male_rate * 100))
    rate_gap.metric('Gap', '{:.2f} points'.format(gap * 100),
                    help='The difference between the male and female disease rates, in percentage points. The wider '
                         'this gap, the sharper the fairness trade-off tends to be.')

    st.divider()
 
    # 7. Fairness metrics panel
 
    # Work out the four metrics from the current predictions and show each one with a pass or fail tag
    metrics = compute_metrics(y_true, predictions, group)
    st.subheader('Fairness Metrics')
    st.caption('Each metric approaches fairness from a different angle, and no single one captures the whole picture. '
               'Three of them are gaps between the two groups, so they read as fair when close to 0, while the '
               'disparate impact ratio compares the groups as a ratio, so it reads as fair when close to 1. For a '
               'reminder of what each one means, hover over the (i) beside it. A metric shows green when its value sits '
               'within the tolerance set in the control panel and red when it falls outside. Because the two groups have '
               'different base rates, bringing one metric closer to fairness will usually push another away, and that '
               'tension is the trade-off at the heart of this dashboard. To see why these gaps arise, the Explanation '
               'tab breaks down how the model uses each feature for the two groups.')
    columns = st.columns(len(metrics))
    for i, name in enumerate(metrics):
        value = metrics[name]
        # Disparate Impact is a ratio that is fair near 1, while the others are differences that are fair near 0
        if name == 'Disparate Impact Ratio':
            within = value >= (1 - tolerance)
        else:
            within = value <= tolerance
        # The help text explains what each metric measures, shown as an (i) tooltip next to the value
        columns[i].metric(name, '{:.4f}'.format(value), help=METRIC_HELP[name])
        if within:
            columns[i].success('Within Tolerance')
        else:
            columns[i].error('Outside Tolerance')
 
    st.divider()
 
    # 8. Performance by sex
 
    # Split recall, precision and the two error rates by sex, so under-diagnosis shows up as a higher female false negative rate
    performance_metrics = {'Recall': recall_score, 'Precision': safe_precision,
                           'False Negative Rate': false_negative_rate, 'False Positive Rate': false_positive_rate}
    performance_by_sex = MetricFrame(metrics=performance_metrics, y_true=y_true, y_pred=predictions,
                                     sensitive_features=group)
 
    # Show recall and precision on the top row and the two error rates below, with the female values on the left and the male values on the right
    st.subheader('Performance by {}'.format(group_label))
    st.caption('The same four measures, broken down separately for female and male patients, so any gap between the '
               'groups becomes visible. Recall, also called the true positive rate, is the share of real disease cases '
               'the model catches. Precision is how often a flag turns out to be correct. The false negative rate is '
               'the share of real cases the model misses, and this is where under-diagnosis shows up, which matters '
               'most in a clinical setting. The false positive rate is the share of healthy patients flagged in error. '
               'For a reminder, hover over the (i) on any of the values.')
    by_group = performance_by_sex.by_group
    female_recall, female_precision, male_recall, male_precision = st.columns(4)
    female_recall.metric('Female Recall', '{:.4f}'.format(by_group['Recall'].values[0]), help=PERF_HELP['Recall'])
    female_precision.metric('Female Precision', '{:.4f}'.format(by_group['Precision'].values[0]), help=PERF_HELP['Precision'])
    male_recall.metric('Male Recall', '{:.4f}'.format(by_group['Recall'].values[1]), help=PERF_HELP['Recall'])
    male_precision.metric('Male Precision', '{:.4f}'.format(by_group['Precision'].values[1]), help=PERF_HELP['Precision'])
 
    female_fnr, female_fpr, male_fnr, male_fpr = st.columns(4)
    female_fnr.metric('Female False Negative Rate', '{:.4f}'.format(by_group['False Negative Rate'].values[0]), help=PERF_HELP['False Negative Rate'])
    female_fpr.metric('Female False Positive Rate', '{:.4f}'.format(by_group['False Positive Rate'].values[0]), help=PERF_HELP['False Positive Rate'])
    male_fnr.metric('Male False Negative Rate', '{:.4f}'.format(by_group['False Negative Rate'].values[1]), help=PERF_HELP['False Negative Rate'])
    male_fpr.metric('Male False Positive Rate', '{:.4f}'.format(by_group['False Positive Rate'].values[1]), help=PERF_HELP['False Positive Rate'])
 
    # Point out the gap in missed cases directly, since this is where under-diagnosis appears
    # The rates are read from the metric frame above, so the sentence updates with the dataset and the thresholds
    female_fnr_value = by_group['False Negative Rate'].values[0]
    male_fnr_value = by_group['False Negative Rate'].values[1]
 
    # The female false negative rate is worked out over the female patients who truly have disease, so its reliability depends on how many there are
    # When that number is small the rate is read as a direction rather than an exact figure, matching how the notebook treats the small female group
    female_true_cases = int(((group == 0) & (y_true == 1)).sum())
    callout = ('At the current thresholds, the model misses **{:.1%}** of the female patients who truly have disease, '
               'against **{:.1%}** of the male patients. A gap like this is exactly where under-diagnosis would surface.'
               .format(female_fnr_value, male_fnr_value))
    if female_true_cases < 30:
        callout += (' That female figure rests on only {} patients who truly have disease, so it is best read as a '
                    'strong signal of direction rather than an exact measurement.'.format(female_true_cases))
    st.info(callout)
 
 
with tab_explain:
 
    # The SHAP explanations were worked out for the baseline model only, so this tab shows a note when a mitigation is chosen
    if mitigation_name != 'Baseline':
        st.info('The explanations on this tab are worked out for the baseline model only, so they appear once the '
                'baseline is selected in the control panel. The SHAP values were saved for the baseline model, since that is '
                'the one being audited, and the mitigations change the decisions rather than the explanation behind '
                'them. Select the baseline to view the explanations, and use the Fairness Metrics, Calibration & ROC, '
                'Errors and Dataset and Metric Comparison tabs to compare the mitigation methods.')
    else:
 
        # 9. Gender-stratified SHAP panel
 
        # Load the gender-stratified SHAP importances for the chosen dataset
        shap_gender = load_shap_gender(config['prefix'])
 
        # Build a sorted importance series for each sex with readable names, so the most important feature sits at the top
        labels = shap_gender['feature'].map(READABLE_LABELS)
        female_importance = pd.Series(data=shap_gender['female'].values, index=labels).sort_values()
        male_importance = pd.Series(data=shap_gender['male'].values, index=labels).sort_values()
 
        # Plot the two rankings side by side, so the way the model uses each feature for each sex can be compared
        st.subheader('Feature Importance by {}'.format(group_label))
        st.caption('This shows which patient details the model relies on most, worked out separately for female and '
                   'male patients using SHAP. Each bar is the average importance of a feature, so the longer the bar, '
                   'the more heavily the model leans on that feature for that group. What to look for is whether sex '
                   'itself sits near the top for one group but not the other. If the model leans on sex when judging '
                   'female patients but on clinical findings when judging male patients, that asymmetry suggests the '
                   'two groups are being treated differently, and it is the clearest link this project draws between '
                   'the explanation and the fairness gap.')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        female_importance.plot(kind='barh', ax=axes[0], color='#e67e22', title='Female Patients')
        male_importance.plot(kind='barh', ax=axes[1], color='#1f3b6f', title='Male Patients')
        axes[0].set_xlabel('Mean Absolute SHAP Value')
        axes[1].set_xlabel('Mean Absolute SHAP Value')
        axes[0].set_ylabel('Feature')
        axes[1].set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Point out the asymmetry where it is present, since it is the clearest sign the two groups are treated differently
        # The sex feature is named sex in one dataset and gender in the other, so its readable label is looked up from the group column
        # The rank is read from the two importance series, so the note fits whichever dataset is shown rather than being fixed
        sex_label = READABLE_LABELS[config['group']]
        female_top_feature = female_importance.index[-1]
        male_ranking = list(male_importance.index[::-1])
        male_sex_rank = male_ranking.index(sex_label) + 1
        if female_top_feature == sex_label and male_sex_rank > 3:
            st.info('Here, {0} is the single most important feature for female patients, yet it ranks only **{1}** for '
                    'male patients. This describes how the model uses {0} rather than pointing to a clinical cause. The '
                    'model leans on {0} when assessing female patients but on the clinical measurements when assessing '
                    'male patients, and that is the pattern of under-diagnosis this project sets out to '
                    'investigate.'.format(group_label.lower(), male_sex_rank))
        else:
            st.info('Look at where {} sits in each of the two rankings. If it is a top feature for one group but not '
                    'the other, that asymmetry suggests the model is treating the two groups differently rather than '
                    'relying on the same clinical findings for both.'.format(group_label.lower()))
 
        # 10. Local explanation panel
 
        # Load the per-patient SHAP values, feature values and base value for the chosen dataset
        shap_values, feature_values, base_value = load_local(config['prefix'])
 
        # Let the user pick one patient by position, so a single prediction can be explained
        st.subheader('Local Explanation for One Patient')
        st.caption('Pick any patient to see why the model reached the decision it did for that individual. The '
                   'waterfall chart begins at the base value, which is the average prediction across the training '
                   'patients, and traces how each feature pushed this patient\'s risk upward towards disease or '
                   'downward towards healthy until it reached the final prediction. Red bars push towards disease and '
                   'blue bars push away from it. The values are given in log-odds, the model\'s internal scale rather '
                   'than probabilities, which is why some of them can be negative. The predicted label and outcome '
                   'below follow the thresholds set in the control panel, using the female or male threshold to match the '
                   'patient, so they update as the sliders are moved. The waterfall itself does not change, since it '
                   'explains the probability rather than the decision.')
        patient = st.number_input('Patient', 0, len(shap_values) - 1, 0)
 
        # Work out what the model predicted for this patient and whether it was a hit, a missed case or a false alarm
        # The threshold comes from the control panel, using the female or male slider to match the patient, so the outcome updates as the sliders move
        patient_threshold = female_threshold if group[patient] == 0 else male_threshold
        patient_label = 1 if proba[patient] >= patient_threshold else 0
        if patient_label == 1 and y_true[patient] == 0:
            error_type = 'False Alarm'
        elif patient_label == 0 and y_true[patient] == 1:
            error_type = 'Missed Case'
        else:
            error_type = 'Correct'
 
        # Show the predicted probability, the predicted and actual labels, the type of error and the sex, side by side
        patient_proba, patient_pred, patient_actual, patient_error, patient_sex = st.columns(5)
        patient_proba.metric('Predicted Probability', '{:.4f}'.format(proba[patient]),
                             help='The model\'s estimated probability that this patient has disease, on a scale from 0 '
                                  'to 1. It is the log-odds output from the waterfall, converted into a probability.')
        patient_pred.metric('Predicted', 'Disease' if patient_label == 1 else 'No Disease',
                            help='The decision the model makes for this patient at the threshold set in the control panel for their group.')
        patient_actual.metric('Actual', 'Disease' if y_true[patient] == 1 else 'No Disease',
                              help='Whether this patient truly has disease according to the data, which is what the '
                                   'prediction is checked against.')
        patient_error.metric('Outcome', error_type,
                             help='A missed case is a patient who has disease but was predicted healthy, the more '
                                  'serious of the two errors. A false alarm is a healthy patient who was predicted to '
                                  'have disease.')
        patient_sex.metric(group_label, 'Female' if group[patient] == 0 else 'Male')
 
        # 11. Local SHAP explanation for the selected patient
 
        # Draw the waterfall for the selected patient, which starts from the base value and shows how each feature moved the prediction
        fig = draw_local_waterfall(patient, shap_values, base_value, feature_values)
        _, middle_column, _ = st.columns([1, 3, 1])
        middle_column.pyplot(fig)
 
        st.divider()
 
 
with tab_errors:
 
    # 12. Missed patients (false negatives)
 
    # Find the patients the model missed with the current predictions, who truly have disease but were predicted healthy
    fn_mask = (y_true == 1) & (predictions == 0)
    all_positions = np.arange(len(y_true))
    female_missed = all_positions[fn_mask & (group == 0)]
    male_missed = all_positions[fn_mask & (group == 1)]
 
    # Work out the false negative rate for each group, so the counts can be shown next to the rate that accounts for the group sizes
    female_true_total = ((group == 0) & (y_true == 1)).sum()
    male_true_total = ((group == 1) & (y_true == 1)).sum()
    female_fnr_display = len(female_missed) / female_true_total if female_true_total > 0 else 0
    male_fnr_display = len(male_missed) / male_true_total if male_true_total > 0 else 0
 
    # Show how many true cases were missed in each group, with the false negative rate alongside so the group sizes are accounted for
    st.subheader('Missed Patients (False Negatives)')
    st.caption('A missed patient truly has disease but was predicted healthy. In a clinical setting this is the more '
               'serious kind of error, since a patient who needs care is sent away believing they are clear. The '
               'counts below are raw numbers, so for a fair comparison between the groups look at the false negative '
               'rate shown beneath each count, or the rates on the Fairness Metrics tab, both of which account for the '
               'different group sizes.')
    female_column, male_column = st.columns(2)
    female_column.metric('Female Missed', len(female_missed))
    female_column.markdown('<span style="color:#6c757d;font-size:0.95rem;">A false negative rate of '
                           '<span style="color:#1f3b6f;font-weight:600;">{:.1%}</span>, out of '
                           '<span style="color:#1f3b6f;font-weight:600;">{}</span> female patients with disease</span>'
                           .format(female_fnr_display, female_true_total), unsafe_allow_html=True)
    male_column.metric('Male Missed', len(male_missed))
    male_column.markdown('<span style="color:#6c757d;font-size:0.95rem;">A false negative rate of '
                         '<span style="color:#1f3b6f;font-weight:600;">{:.1%}</span>, out of '
                         '<span style="color:#1f3b6f;font-weight:600;">{}</span> male patients with disease</span>'
                         .format(male_fnr_display, male_true_total), unsafe_allow_html=True)
 
    # List every missed patient with their sex, and add the probability unless the threshold optimiser is chosen, since it gives no probability
    missed = all_positions[fn_mask]
    missed_sex = []
    for position in missed:
        missed_sex.append('Female' if group[position] == 0 else 'Male')
    if optimiser_chosen:
        # Tell the user why the probability column is missing, since the threshold optimiser returns a decision rather than a score
        st.caption('The threshold optimiser produces a decision directly rather than a probability score, so the predicted-probability column is not shown for it.')
        missed_table = pd.DataFrame({'Position': missed, group_label: missed_sex})
    else:
        missed_table = pd.DataFrame({'Position': missed, group_label: missed_sex, 'Predicted Probability': proba[missed]})
 
    # Let the user filter the table by sex, so the missed cases of one group can be looked at on their own
    missed_filter = st.selectbox('Filter Missed Patients by {}'.format(group_label), ['All', 'Female', 'Male'])
    if missed_filter != 'All':
        missed_table = missed_table[missed_table[group_label] == missed_filter]
    st.dataframe(missed_table.round(4), hide_index=True)
 
    st.divider()
 
    # 13. False alarms (false positives)
 
    # Find the false alarms with the current predictions, who are truly healthy but were predicted to have disease
    fp_mask = (y_true == 0) & (predictions == 1)
    false_alarm = all_positions[fp_mask]
    female_alarm = all_positions[fp_mask & (group == 0)]
    male_alarm = all_positions[fp_mask & (group == 1)]
 
    # Work out the false positive rate for each group, so the counts can be shown next to the rate that accounts for the group sizes
    female_healthy_total = ((group == 0) & (y_true == 0)).sum()
    male_healthy_total = ((group == 1) & (y_true == 0)).sum()
    female_fpr_display = len(female_alarm) / female_healthy_total if female_healthy_total > 0 else 0
    male_fpr_display = len(male_alarm) / male_healthy_total if male_healthy_total > 0 else 0
 
    # Show how many healthy patients were flagged in each group, with the false positive rate alongside so the group sizes are accounted for
    st.subheader('False Alarms (False Positives)')
    st.caption('A false alarm flags a healthy patient as having disease. This is less serious than a missed case, but '
               'it still leads to unnecessary tests and needless worry. The counts below are raw numbers, so for a fair '
               'comparison between the groups look at the false positive rate shown beneath each count, or the rates on '
               'the Fairness Metrics tab, both of which account for the different group sizes.')
    female_alarm_column, male_alarm_column = st.columns(2)
    female_alarm_column.metric('Female False Alarms', len(female_alarm))
    female_alarm_column.markdown('<span style="color:#6c757d;font-size:0.95rem;">A false positive rate of '
                                 '<span style="color:#1f3b6f;font-weight:600;">{:.1%}</span>, out of '
                                 '<span style="color:#1f3b6f;font-weight:600;">{}</span> healthy female patients</span>'
                                 .format(female_fpr_display, female_healthy_total), unsafe_allow_html=True)
    male_alarm_column.metric('Male False Alarms', len(male_alarm))
    male_alarm_column.markdown('<span style="color:#6c757d;font-size:0.95rem;">A false positive rate of '
                               '<span style="color:#1f3b6f;font-weight:600;">{:.1%}</span>, out of '
                               '<span style="color:#1f3b6f;font-weight:600;">{}</span> healthy male patients</span>'
                               .format(male_fpr_display, male_healthy_total), unsafe_allow_html=True)
 
    # List every false alarm with their sex, and add the probability unless the threshold optimiser is chosen, since it gives no probability
    alarm_sex = []
    for position in false_alarm:
        alarm_sex.append('Female' if group[position] == 0 else 'Male')
    if optimiser_chosen:
        # Tell the user why the probability column is missing, since the threshold optimiser returns a decision rather than a score
        st.caption('The threshold optimiser produces a decision directly rather than a probability score, so the predicted-probability column is not shown for it.')
        alarm_table = pd.DataFrame({'Position': false_alarm, group_label: alarm_sex})
    else:
        alarm_table = pd.DataFrame({'Position': false_alarm, group_label: alarm_sex, 'Predicted Probability': proba[false_alarm]})
 
    # Let the user filter the table by sex, so the false alarms of one group can be looked at on their own
    alarm_filter = st.selectbox('Filter False Alarms by {}'.format(group_label), ['All', 'Female', 'Male'])
    if alarm_filter != 'All':
        alarm_table = alarm_table[alarm_table[group_label] == alarm_filter]
    st.dataframe(alarm_table.round(4), hide_index=True)
 
 
with tab_quality:
 
    # 14. Calibration by sex
 
    # Calibration is the fifth fairness metric, and the threshold optimiser has no probability, so the panel shows a note instead when it is chosen
    st.subheader('Calibration by {}'.format(group_label))
    st.caption('The fifth fairness metric asks whether a predicted risk score means the same thing for female and '
               'male patients. A model is well calibrated when, among the patients given a risk of around 0.7, about '
               '70 percent truly have disease, and this should hold for both groups. A curve on the diagonal means the '
               'scores can be trusted; a curve below it means the model overestimates the risk for that group, and a '
               'curve above it means it underestimates. The curve depends only on the probability scores, so it does '
               'not move when the sliders are changed.')
    if optimiser_chosen:
        st.info('The threshold optimiser produces a decision directly rather than a probability score, and since a '
                'calibration curve needs a score to plot, there is none to show for this method. Select the baseline or '
                'one of the two pre-processing mitigations to see the calibration.')
    else:
        # Load the calibration curve points for the chosen dataset and mitigation method
        female_calibration, male_calibration = load_calibration(config['prefix'], CALIBRATION_SUFFIX[mitigation_name])
 
        # Plot the reliability diagram for each sex against the diagonal, so a curve near the diagonal means the probabilities can be trusted
        fig = plt.figure(figsize=(5, 4))
        plt.plot(female_calibration['prob_pred'], female_calibration['prob_true'], marker='o', color='#e67e22', label='Female')
        plt.plot(male_calibration['prob_pred'], male_calibration['prob_true'], marker='o', color='#1f3b6f', label='Male')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Observed Proportion of True Cases')
        plt.legend()
        plt.tight_layout()
 
        # Put the chart in a centred column so it sits in the middle at about the width of one SHAP chart
        _, middle_column, _ = st.columns([1, 2, 1])
        middle_column.pyplot(fig)
 
    st.divider()
 
    # 15. ROC by sex
 
    # The threshold optimiser has no probability to rank the patients by, so there is no ROC curve and the panel shows a note instead
    st.subheader('ROC by {}'.format(group_label))
    st.caption('This shows how well the model separates disease cases from healthy patients within each group, across '
               'every threshold. The closer a curve runs to the top-left the better, and the AUC sums this up in a '
               'single score that runs from 0.5, no better than chance, up to 1, a perfect separation. When the two '
               'curves are close, the model ranks both groups equally well, so a fairness gap in the earlier metrics is '
               'not because the model separates one group\'s cases worse, but because the single shared threshold falls '
               'at a different point for each group. This is why the gap points towards adjusting the threshold per '
               'group rather than retraining the model. ROC is not one of the five fairness criteria measured against '
               'the tolerance; it is here to help show where the equalised odds gap comes from. Like the calibration '
               'curve, it uses only the probability scores, so it does not change when the sliders are moved.')
    if optimiser_chosen:
        st.info('The threshold optimiser produces a decision directly rather than a probability score, and since an ROC '
                'curve needs a score to rank patients by, there is none to show for this method. Select the baseline or '
                'one of the two pre-processing mitigations to see the ROC.')
    else:
        # Compute the ROC curve for each sex from the chosen probabilities, so the two groups can be compared across every threshold
        female_fpr, female_tpr, _ = roc_curve(y_true[group == 0], proba[group == 0])
        male_fpr, male_tpr, _ = roc_curve(y_true[group == 1], proba[group == 1])
 
        # Plot the two curves against the diagonal, so a curve further above the diagonal means the model ranks that group better
        fig = plt.figure(figsize=(5, 4))
        plt.plot(female_fpr, female_tpr, color='#e67e22', label='Female (AUC = {:.4f})'.format(roc_auc_score(y_true[group == 0], proba[group == 0])))
        plt.plot(male_fpr, male_tpr, color='#1f3b6f', label='Male (AUC = {:.4f})'.format(roc_auc_score(y_true[group == 1], proba[group == 1])))
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
 
        # Put the chart in a centred column so it sits in the middle like the calibration plot
        _, roc_column, _ = st.columns([1, 2, 1])
        roc_column.pyplot(fig)
 
 
with tab_compare:
 
    # 16. Comparing the two datasets
 
    # Load the probabilities and base rates for both datasets, so the two can be compared side by side at the current thresholds
    uci_probabilities, uci_base_rate = load_data('uci')
    kaggle_probabilities, kaggle_base_rate = load_data('kaggle')
 
    # Show the base rate gap for each dataset, which is the cause of how large the fairness gaps are
    st.subheader('Disease Rate Gap by Dataset')
    st.caption('This is the gap in disease rate between male and female patients within each dataset. The UCI data has '
               'a wide gap while the Kaggle data has almost none, and that contrast is the natural experiment at the '
               'heart of the project. Because the difficulty of satisfying every fairness measure at once is driven by '
               'the base-rate gap, the UCI data shows sharp trade-offs while the Kaggle data shows barely any. This gap '
               'is a property of the data itself, so it does not move when the sliders are changed.')
    uci_gap = (uci_base_rate['disease_rate'].values[1] - uci_base_rate['disease_rate'].values[0]) * 100
    kaggle_gap = (kaggle_base_rate['disease_rate'].values[1] - kaggle_base_rate['disease_rate'].values[0]) * 100
    uci_gap_column, kaggle_gap_column = st.columns(2)
    uci_gap_column.metric('UCI Heart Disease', '{:.2f} points'.format(uci_gap),
                          help='The base-rate gap for this dataset, in percentage points. A wide gap makes the fairness '
                               'trade-off sharp, while a near-zero gap keeps it mild.')
    kaggle_gap_column.metric('Kaggle Cardiovascular Disease', '{:.2f} points'.format(kaggle_gap),
                             help='The base-rate gap for this dataset, in percentage points. A wide gap makes the '
                                  'fairness trade-off sharp, while a near-zero gap keeps it mild.')
 
    st.divider()
 
    # List the methods and metrics the user can pick from, so the comparison can be shown for one or all of them
    comparison_methods = ['Baseline', 'SMOTE-NC', 'Reweighting', 'Threshold Optimiser']
    comparison_metrics = ['Demographic Parity Difference', 'Equalised Odds Difference',
                          'Predictive Parity Difference', 'Disparate Impact Ratio', 'Accuracy', 'Recall']
 
    # The difference metrics are fair near 0, so a tolerance line is drawn for them, while the ratio is fair near 1 and gets a parity line instead
    difference_metrics = ['Demographic Parity Difference', 'Equalised Odds Difference', 'Predictive Parity Difference']
 
    # The direction that counts as better differs by metric, so the note under the chart is chosen to match the metric on show
    direction_help = {'Demographic Parity Difference': 'A lower value is better here, since a difference close to 0 means the two groups are treated more equally.',
                      'Equalised Odds Difference': 'A lower value is better here, since a difference close to 0 means the two groups are treated more equally.',
                      'Predictive Parity Difference': 'A lower value is better here, since a difference close to 0 means the two groups are treated more equally.',
                      'Disparate Impact Ratio': 'A ratio closer to 1 is better, and a value below 0.8 is the common threshold for concern.',
                      'Accuracy': 'A higher value is better, since this measures how often the model is correct overall.',
                      'Recall': 'A higher value is better, since this measures how many true disease cases the model catches.'}
 
    # Work out every method's metrics for both datasets at the current thresholds, so the panels below read from live results
    uci_results = {}
    kaggle_results = {}
    for method in comparison_methods:
        uci_results[method] = metrics_for_method(uci_probabilities, 'sex', method, female_threshold, male_threshold)
        kaggle_results[method] = metrics_for_method(kaggle_probabilities, 'gender', method, female_threshold, male_threshold)
 
    # Let the user pick which metric to compare, with an option to see every metric at once, and whether to see every method or just one
    st.subheader('Fairness and Performance Results')
    st.caption('Pick a single metric to compare the two datasets as bars, or pick All to read every metric together in '
               'the table below. These values are worked out at the current slider thresholds, so they move as the '
               'sliders are adjusted, which makes it possible to see how the same change plays out across both datasets '
               'at once.')
    metric_choice = st.selectbox('Metric', ['All'] + comparison_metrics)
    method_choice = st.selectbox('Method', ['All'] + comparison_methods)
 
    # Flag the threshold optimiser's behaviour clearly here, since it sets its own decision and so does not move with the sliders like the other methods
    st.warning('The threshold optimiser is the exception among the methods: it sets its own threshold for each sex, so '
               'its results stay fixed as the sliders are moved rather than responding to them.')
 
    # A single metric is drawn as a bar chart, while all metrics together show a note instead, since they sit on different scales
    if metric_choice == 'All':
        st.info('When All is selected the metrics appear only in the table below, since they sit on different scales, '
                'some of them gaps near 0 and one a ratio near 1, and would not be readable together on a single '
                'chart. Pick a single metric to see it drawn as bars.')
    else:
        # Show the direction that counts as better for the chosen metric, so a tall bar is not read as good or bad by mistake
        st.caption(direction_help[metric_choice])
 
        # When a single method is chosen, compare the two datasets on that one method, otherwise compare all four methods side by side
        if method_choice == 'All':
            uci_values = []
            kaggle_values = []
            for method in comparison_methods:
                uci_values.append(uci_results[method][metric_choice])
                kaggle_values.append(kaggle_results[method][metric_choice])
 
            # Draw the four methods as grouped bars, with one bar for each dataset, so the effect of each method can be read off
            positions = np.arange(len(comparison_methods))
            width = 0.35
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(positions - width / 2, uci_values, width, color='#7b241c', label='UCI Heart Disease')
            ax.bar(positions + width / 2, kaggle_values, width, color='#999999', label='Kaggle Cardiovascular Disease')
            ax.set_xticks(positions)
            ax.set_xticklabels(comparison_methods)
            ax.set_ylabel(metric_choice)
 
            # Add the fairness reference line, so the bars can be read against the level that counts as fair
            # A difference metric is fair below the tolerance, while the ratio is fair at 1 with the four-fifths threshold at 0.8
            if metric_choice in difference_metrics:
                ax.axhline(tolerance, linestyle='--', color='#999999', label='Tolerance ({:.2f})'.format(tolerance))
            elif metric_choice == 'Disparate Impact Ratio':
                ax.axhline(1.0, linestyle='--', color='#999999', label='Parity (1.0)')
                ax.axhline(0.8, linestyle=':', color='#999999', label='Four-fifths (0.8)')
 
            # Put the legend above the plot so it does not sit on top of the bars
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
            plt.tight_layout()
        else:
            uci_value = uci_results[method_choice][metric_choice]
            kaggle_value = kaggle_results[method_choice][metric_choice]
 
            # Draw a single bar for each dataset, so the two can be compared directly for the chosen method
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(['UCI Heart Disease', 'Kaggle Cardiovascular Disease'], [uci_value, kaggle_value],
                   color=['#7b241c', '#999999'])
            ax.set_ylabel(metric_choice)
 
            # Add the fairness reference line for this metric, matching the grouped view, so the single bars can also be read against it
            if metric_choice in difference_metrics:
                ax.axhline(tolerance, linestyle='--', color='#999999', label='Tolerance ({:.2f})'.format(tolerance))
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
            elif metric_choice == 'Disparate Impact Ratio':
                ax.axhline(1.0, linestyle='--', color='#999999', label='Parity (1.0)')
                ax.axhline(0.8, linestyle=':', color='#999999', label='Four-fifths (0.8)')
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
            plt.tight_layout()
 
        # Put the chart in a centred column so it sits in the middle of the page
        _, middle_column, _ = st.columns([1, 3, 1])
        middle_column.pyplot(fig)
 
    # Decide which metrics and methods the table below should show, following the current choices
    if metric_choice == 'All':
        table_metrics = comparison_metrics
    else:
        table_metrics = [metric_choice]
    if method_choice == 'All':
        table_methods = comparison_methods
    else:
        table_methods = [method_choice]
 
    # Read the chosen values into a table with a two-level header, so the method sits on top and the dataset sits below it
    table_columns = []
    table_series = []
    for method in table_methods:
        uci_column = []
        kaggle_column = []
        for metric in table_metrics:
            uci_column.append(uci_results[method][metric])
            kaggle_column.append(kaggle_results[method][metric])
        table_columns.append((method, 'UCI'))
        table_series.append(uci_column)
        table_columns.append((method, 'Kaggle'))
        table_series.append(kaggle_column)
 
    # Build the frame with the paired columns, then give it the two-level header so the dataset sits on a second row
    comparison_table = pd.DataFrame(dict(zip(range(len(table_series)), table_series)), index=table_metrics)
    comparison_table.columns = pd.MultiIndex.from_tuples(table_columns, names=['Method', 'Dataset'])
    st.dataframe(comparison_table.round(4))
