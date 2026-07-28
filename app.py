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
 
# Matplotlib for the panels that draw charts, from the SHAP importances to the map of workable threshold pairs
from matplotlib import pyplot as plt
 
# A two-colour map for the panel that shades the threshold pairs meeting the tolerance
from matplotlib.colors import ListedColormap
 
# Use the full width of the browser, since the dashboard has several panels side by side
st.set_page_config(page_title='Fairness Trade-off Dashboard', layout='wide')
 
# Theme colours and background are set in .streamlit/config.toml; this block only adds the fine details config.toml cannot reach, such as the metric sizing, the card shadows, the heading styling, the tab styling, the expander styling and the dropdown styling
st.markdown('''
    <style>
    /* Shrink the large metric numbers to roughly half their default size, and keep them bold so the values stand out from the labels and captions around them */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    /* Keep the metric label readable alongside the smaller value */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    /* Set the caption text to a clear, readable size for the panel explanations */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        font-size: 0.90rem;
    }
    /* Give each metric a white card look with a soft border and shadow, so the boxes lift off the grey background */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 3px rgba(31, 59, 111, 0.06);
    }
    /* Style the page title in navy with a thin underline, so it reads as a clear branded header */
    h1 {
        color: #1f3b6f !important;
        font-size: 2.0rem !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }
    /* Mark each panel heading in navy with a short left bar, with a balanced gap to the text and room below so the caption underneath is not cramped */
    h2, h3 {
        color: #1f3b6f !important;
        font-size: 1.4rem !important;
        border-left: 4px solid #1f3b6f;
        padding-left: 0.5rem !important;
        margin-left: 0;
        margin-bottom: 0.6rem !important;
    }
    /* Sit the tab strip in a light grey band with spacing between the tabs, so it is clear they are separate sections that can be switched */
    .stTabs [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        background-color: #eef2f7;
        padding: 0.4rem;
        border-radius: 0.6rem;
    }
    /* Give each tab a white button look with a border, so the sections are easy to tell apart and read as clickable */
    .stTabs [data-baseweb="tab"],
    [data-testid="stTab"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    /* Set the tab label size and weight on the inner text element, since the label sits inside a paragraph within the tab */
    .stTabs [data-baseweb="tab"] p,
    [data-testid="stTab"] p {
        font-size: 1rem;
        font-weight: 600;
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
    /* Give each expander the same white card look as the metrics, with a navy outline so the closed panels read as sections that open */
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #1f3b6f;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(31, 59, 111, 0.06);
    }
    /* Drop the border Streamlit puts on the inner element, so only the outer navy outline shows */
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] > div {
        border: none;
        background-color: transparent;
    }
    /* Set the expander heading in navy, one step below the tab labels so the panels read as sections inside the tab rather than beside it */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #1f3b6f;
        font-weight: 600;
        font-size: 0.95rem;
    }
    /* Match the arrow to the heading colour, since it sits inside the summary as a separate icon */
    [data-testid="stExpander"] summary svg {
        fill: #1f3b6f;
    }
    /* Give the dropdown controls the same border as the metric cards, so the boxes on the page all read as one set */
    [data-testid="stSelectbox"] div[role="group"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border: 1px solid #cbd5e1 !important;
    }
    /* Set the selected text inside the dropdowns to the body size, so the filter boxes on the main tabs do not sit larger than the text around them */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        font-size: 0.90rem !important;
    }
    /* Set the options in the open dropdown list to the same size, since the list opens in a separate layer that the rule above does not reach */
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li div,
    div[data-baseweb="popover"] li span,
    ul[role="listbox"] li,
    div[role="listbox"] li {
        font-size: 0.90rem !important;
    }
    /* Set the text in the coloured status boxes below the caption size, so the green, red, amber and blue messages do not sit larger than the panel text around them */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
        font-size: 0.90rem !important;
    }
    /* Set the body paragraphs a touch smaller, so the task description under the title reads as supporting text rather than competing with the headings */
    [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem;
    }
    /* Set the control panel heading to the same size as the panel headings on the main page, so the sidebar reads as a section of equal weight */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.4rem !important;
    }
    /* Keep the control labels a balanced step below that heading, so the controls read the same across Streamlit versions rather than shrinking on the deployed one */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        font-size: 0.85rem;
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
 
# The plain-language meaning of each fairness metric, shown as a help tooltip next to it on the fairness metrics panel
METRIC_HELP = {
    'Demographic Parity Difference': 'Whether the model flags disease at the same rate for both sexes. It measures the '
                                     'gap between the proportion of female patients and the proportion of male patients '
                                     'predicted to have disease, so a value of 0 means both groups are flagged equally '
                                     'often.',
    'Equalised Odds Difference': 'Whether the model catches disease and raises false alarms at the same rate for both '
                                 'sexes, looking separately at the patients who truly have disease and those who do '
                                 'not. It takes the larger of two gaps, the gap in the true positive rate and the gap '
                                 'in the false positive rate, so a value of 0 means both rates match for each group.',
    'Predictive Parity Difference': 'Whether a positive prediction can be trusted equally for both sexes. It is the gap '
                                    'in precision between the groups, precision being the proportion of flagged patients '
                                    'who truly have disease, so a value of 0 means a \'disease\' flag carries the same '
                                    'weight for a female and a male patient.',
    'Disparate Impact Ratio': 'The ratio between the lower and the higher of the two groups\' positive-prediction '
                              'rates, so it runs from 0 to 1. A ratio of 1 means both groups are flagged equally often. '
                              'The 0.8 line (the four-fifths rule) is shown in the comparison view as a common '
                              'reference point.'}
 
# The plain-language meaning of each performance measure, shown as a help tooltip next to it on the performance panel
PERF_HELP = {
    'Recall': 'Among the patients who truly have disease, the proportion the model correctly flags. A higher value is '
              'better, since it means fewer real cases are missed. It is also known as the true positive rate, or '
              'sensitivity.',
    'Precision': 'Among the patients the model flags as having disease, the proportion who truly have disease. A higher '
                 'value is better, since it means a flag is more often correct.',
    'False Negative Rate': 'Among the patients who truly have disease, the proportion the model predicts healthy. These '
                           'are missed cases, and since a missed case is usually more serious than a false alarm in a '
                           'clinical setting, a lower value is preferable.',
    'False Positive Rate': 'Among the healthy patients, the proportion the model wrongly flags as having disease. These '
                           'are false alarms, which lead to unnecessary tests, so a lower value is preferable.'}
 
# The description of every feature the model uses, shown as a table on the overview tab so each column can be understood
FEATURE_INFO = {
    'uci': [('Age', 'Age of the patient in years'),
            ('Sex', 'Sex of the patient (0 = female, 1 = male)'),
            ('Chest Pain', 'Chest pain type (0 = asymptomatic, 1 = atypical angina, 2 = non-anginal, 3 = typical angina)'),
            ('Blood Pressure', 'Resting blood pressure on admission, in mm Hg'),
            ('Cholesterol', 'Serum cholesterol in mg/dl'),
            ('Fasting Blood Sugar', 'Whether fasting blood sugar is greater than 120 mg/dl (0 = false, 1 = true)'),
            ('Resting ECG', 'Resting electrocardiographic result (0 = LV hypertrophy, 1 = normal, 2 = ST-T abnormality)'),
            ('Max Heart Rate', 'Maximum heart rate achieved'),
            ('Exercise Angina', 'Whether exercise induced angina (0 = false, 1 = true)'),
            ('ST Depression', 'ST depression induced by exercise relative to rest')],
    'kaggle': [('Age', 'Age of the patient, converted from days to years'),
               ('Gender', 'Gender of the patient (0 = women, 1 = men)'),
               ('Height', 'Height of the patient in centimetres'),
               ('Weight', 'Weight of the patient in kilograms'),
               ('Systolic BP', 'Systolic blood pressure'),
               ('Diastolic BP', 'Diastolic blood pressure'),
               ('Cholesterol', 'Cholesterol level (1 = normal, 2 = above normal, 3 = well above normal)'),
               ('Glucose', 'Glucose level (1 = normal, 2 = above normal, 3 = well above normal)'),
               ('Smoking', 'Whether the patient smokes (0 = no, 1 = yes)'),
               ('Alcohol', 'Whether the patient drinks alcohol (0 = no, 1 = yes)'),
               ('Active', 'Whether the patient is physically active (0 = no, 1 = yes)')]}
 
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
 
 
# 3. Metric calculation and the drawing helpers
 
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
 
# Draw a rate as a hundred squares, since a count of people is easier to judge than a decimal
def icon_array(highlighted, highlight_colour, base_colour):
    """Build a ten by ten grid of squares with the first ones shaded, so a rate can be read as a count of people."""
    squares = []
    for position in range(100):
        colour = highlight_colour if position < highlighted else base_colour
        squares.append('<div style="width:12px;height:12px;background:{};border-radius:2px;"></div>'.format(colour))
    return '<div style="display:grid;grid-template-columns:repeat(10,12px);gap:3px;">{}</div>'.format(''.join(squares))
 
# Work out the four fairness measures at every pair of thresholds, so the panel can search rather than leaving the user to hunt
# The measures do not depend on the tolerance, so the grid is cached once and only the comparison is redone when the slider moves
@st.cache_data
def threshold_grid(prefix, group_column, method, step=0.01):
    """Return the four fairness measures at every pair of thresholds."""
    probabilities, _ = load_data(prefix)
    group = probabilities[group_column].values
    y_true = probabilities['y_true'].values
    proba = probabilities[MITIGATIONS[method]].values
    grid = np.arange(0, 1 + step / 2, step)
 
    female_scores = proba[group == 0]
    male_scores = proba[group == 1]
    female_truth = y_true[group == 0]
    male_truth = y_true[group == 1]
 
    # Each row holds the flags at one threshold, which lets the whole grid be worked out without looping over it
    female_flags = female_scores[None, :] >= grid[:, None]
    male_flags = male_scores[None, :] >= grid[:, None]
 
    female_selected = female_flags.mean(axis=1)
    male_selected = male_flags.mean(axis=1)
    female_caught = female_flags[:, female_truth == 1].mean(axis=1)
    male_caught = male_flags[:, male_truth == 1].mean(axis=1)
    female_alarmed = female_flags[:, female_truth == 0].mean(axis=1)
    male_alarmed = male_flags[:, male_truth == 0].mean(axis=1)
 
    # A threshold that flags nobody has no precision to report, so those rows are held at 0 rather than dividing by zero
    female_flagged = female_flags.sum(axis=1)
    male_flagged = male_flags.sum(axis=1)
    female_caught_count = female_flags[:, female_truth == 1].sum(axis=1)
    male_caught_count = male_flags[:, male_truth == 1].sum(axis=1)
    female_precision = np.zeros(len(grid))
    male_precision = np.zeros(len(grid))
    female_has_flag = female_flagged > 0
    male_has_flag = male_flagged > 0
    female_precision[female_has_flag] = female_caught_count[female_has_flag] / female_flagged[female_has_flag]
    male_precision[male_has_flag] = male_caught_count[male_has_flag] / male_flagged[male_has_flag]

    # The female threshold runs down the rows and the male threshold across the columns, so every cell is one pair
    demographic = np.abs(female_selected[:, None] - male_selected[None, :])
    equalised = np.maximum(np.abs(female_caught[:, None] - male_caught[None, :]),
                           np.abs(female_alarmed[:, None] - male_alarmed[None, :]))
    predictive = np.abs(female_precision[:, None] - male_precision[None, :])
    larger = np.maximum(female_selected[:, None], male_selected[None, :])
    smaller = np.minimum(female_selected[:, None], male_selected[None, :])
    # A pair where nobody is flagged in either group has no ratio to report, so those cells are held at 0
    impact = np.zeros_like(larger)
    has_larger = larger > 0
    impact[has_larger] = smaller[has_larger] / larger[has_larger]
    return demographic, equalised, predictive, impact
 
 
# 4. Sidebar controls and page layout
 
# State who the dashboard is for and what task it supports, so the panels below can be read against a clear purpose
st.title('Fairness Trade-off Dashboard')
st.write('This dashboard is for the person who trains a clinical prediction model and has to decide whether its '
         'behaviour is acceptable before handing it to a clinician. The user sets how large a difference between '
         'the two patient groups is acceptable, and the dashboard shows whether any combination of the two decision '
         'thresholds meets those constraints. The two heart disease datasets are worked examples, and the same '
         'approach applies to any binary clinical prediction task with a sensitive attribute.')
st.write('The dashboard does not pick a fair model on the user\'s behalf. When the two groups have different '
         'underlying disease rates, no setting satisfies every fairness measure at once, so improving one comes '
         'at the cost of another. That trade-off is what the dashboard lays out. It makes this a judgement, not a '
         'calculation, so the dashboard reports each measure in terms of patients rather than leaving it as a '
         'decimal.')
 
# List the three mitigation methods on their own, since the baseline is the untouched model rather than a mitigation
mitigation_methods = ['SMOTE-NC', 'Reweighting', 'Threshold Optimiser']
 
# Put all the controls in the control panel, so they stay in one place while the user moves between the tabs
with st.sidebar:
    st.header('Controls')
    dataset_name = st.selectbox('Dataset', list(DATASETS.keys()),
                                help='Choose which dataset to explore. Every panel updates to reflect the one selected here.')
 
    # Work out the prefix now, since the panels below load their data by it
    config = DATASETS[dataset_name]
    prefix = config['prefix']
 
    # Keep the baseline separate from the mitigations, since it is the untouched model that the mitigations are compared against
    use_mitigation = st.radio('Model', ['Baseline', 'Apply a mitigation'],
                              help='The baseline is the tuned XGBoost model before any bias mitigation is applied. A mitigation is a method that tries to make it fairer.')
    if use_mitigation == 'Apply a mitigation':
        mitigation_name = st.radio('Mitigation Method', mitigation_methods,
                                   help='SMOTE-NC adds synthetic minority-class patients and reweighting changes the '
                                        'weight each patient carries in training, and both retrain the model, which '
                                        'changes the probability scores. The threshold optimiser leaves the model '
                                        'untouched and simply picks a separate threshold for each sex, so the scores '
                                        'themselves stay the same.')
    else:
        mitigation_name = 'Baseline'
 
    # The threshold optimiser sets its own threshold for each sex, so the sliders are switched off when it is chosen
    optimiser_chosen = mitigation_name == 'Threshold Optimiser'
    female_threshold = st.slider('Female Threshold', 0.0, 1.0, 0.5, 0.01, disabled=optimiser_chosen,
                                 help='The threshold used for female patients. A patient is flagged as having disease '
                                      'when the model\'s score reaches it or goes above. The usual starting point is '
                                      '0.50, meaning a patient is flagged once the model sees at least an even chance of '
                                      'disease. Lowering the threshold flags more of them, raising it flags fewer.')
    male_threshold = st.slider('Male Threshold', 0.0, 1.0, 0.5, 0.01, disabled=optimiser_chosen,
                               help='The threshold used for male patients. A patient is flagged as having disease when '
                                    'the model\'s score reaches it or goes above. The usual starting point is 0.50, '
                                    'meaning a patient is flagged once the model sees at least an even chance of disease. '
                                    'Lowering the threshold flags more of them, raising it flags fewer.')
    tolerance = st.slider('Tolerance', 0.0, 0.5, 0.1, 0.01,
                          help='How large a gap between the two groups still counts as fair. On the Fairness Metrics '
                               'tab a metric turns green when it sits within this tolerance and red when it falls '
                               'outside. Moving the slider shows which metrics pass under a strict bar and which only '
                               'pass under a looser one.')
 
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
 
# The word for the sensitive attribute differs by dataset, so it is set once here and reused in the headings and notes below
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
 
# The task needs two panels, and the rest is the data and the analysis behind them, so the strip carries the task and holds the rest behind one tab
st.write('**The task sits in the first two tabs:** Fairness Metrics reports the four measures against the tolerance, '
         'and Errors shows the patients behind those numbers. The data and analysis behind them sit in the third tab.')
tab_tradeoff, tab_errors, tab_analysis = st.tabs(['Fairness Metrics', 'Errors', 'Data and Analysis'])
 
# The four panels open one at a time inside their own tab, so a second strip of tabs does not compete with the one above
with tab_analysis:
    panel_overview = st.expander('Dataset Overview')
    panel_explain = st.expander('Explanation')
    panel_quality = st.expander('Calibration & ROC')
    panel_compare = st.expander('Dataset and Metric Comparison')
 
with tab_tradeoff:
 
    # 5. Settings that meet the tolerance
 
    # The task is to find a pair of thresholds that meets the fairness bar, so the panel searches every pair rather than leaving the user to hunt
    st.subheader('Settings That Meet the Tolerance')
    st.caption('A setting is a pair of thresholds, one threshold for female patients and one for male patients. Every '
               'point on this map is a pair, shaded when all four measures sit within the tolerance set in the control '
               'panel. Tightening the tolerance shrinks the shaded area, and where the two groups have very different '
               'disease rates it disappears altogether. The base rate panel under Data and Analysis gives the size of '
               'that gap. The marker shows where the sliders currently sit.')
 
    # The feasible region map is only drawn for the threshold methods, since the optimiser sets its own threshold and there is no pair to search
    if optimiser_chosen:
        st.info('The threshold optimiser chooses its own threshold for each sex, so there is no pair for the user to '
                'set and nothing to search. The decision summary below reports how its result sits against the '
                'tolerance.')
    else:
        demographic, equalised, predictive, impact = threshold_grid(prefix, config['group'], mitigation_name)
        meets = ((demographic <= tolerance) & (equalised <= tolerance) &
                 (predictive <= tolerance) & (impact >= 1 - tolerance))
 
        # The shaded cells are the settings that pass, with the female threshold down the rows and the male threshold across
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(meets, origin='lower', extent=[0, 1, 0, 1], aspect='auto',
                  cmap=ListedColormap(['#e2e8f0', '#1e6b3a']))
        ax.plot(male_threshold, female_threshold, marker='o', markersize=6, color='#7b241c')
        ax.set_xlabel('Male Threshold', fontsize=9)
        ax.set_ylabel('Female Threshold', fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        _, middle_column, _ = st.columns([1, 2, 1])
        middle_column.pyplot(fig)
        plt.close(fig)
 
        # Say straight out whether the shaded area exists
        working = int(meets.sum())
        if working > 0:
            st.success('**Some settings meet all four measures at a tolerance of {:.2f}.** They are the shaded area on '
                       'the map, and the marker shows where the current sliders sit.'.format(tolerance))
        else:
            st.error('**No setting meets all four measures at a tolerance of {:.2f}.** Loosening the tolerance or '
                     'accepting a gap on at least one measure is the only way through.'.format(tolerance))
 
    # Say straight out whether the current result passes, and if it does not, name the measures it fails, so it does not have to be read off the map or the cards
    # This runs for every method, so the threshold optimiser gets the same decision summary even though it has no pair on the map
    current = compute_metrics(y_true, predictions, group)
    current_fails = []
    if current['Demographic Parity Difference'] > tolerance:
        current_fails.append('Demographic Parity')
    if current['Equalised Odds Difference'] > tolerance:
        current_fails.append('Equalised Odds')
    if current['Predictive Parity Difference'] > tolerance:
        current_fails.append('Predictive Parity')
 
    # An extreme threshold can flag nobody in a group, which leaves the ratio undefined, so an undefined ratio counts as a fail rather than slipping through as a pass
    di_value = current['Disparate Impact Ratio']
    if not np.isfinite(di_value) or di_value < 1 - tolerance:
        current_fails.append('the Disparate Impact Ratio')
 
    # The threshold optimiser sets its own threshold, so its summary drops the slider values that the other methods report
    if len(current_fails) == 0:
        if optimiser_chosen:
            st.success('**Threshold optimiser result: it meets all four measures.**')
        else:
            st.success('**Current setting: it meets all four measures.** The female threshold is {:.2f} and the male '
                       'threshold is {:.2f}.'.format(female_threshold, male_threshold))
    else:
        # Join the failing measures into readable English, since a list joined with 'and' between every item reads badly
        if len(current_fails) == 1:
            failed_text = current_fails[0]
        elif len(current_fails) == 2:
            failed_text = '{} and {}'.format(current_fails[0], current_fails[1])
        else:
            failed_text = '{}, and {}'.format(', '.join(current_fails[:-1]), current_fails[-1])
        if optimiser_chosen:
            st.warning('**Threshold optimiser result: it does not meet {}.**'.format(failed_text))
        else:
            st.warning('**Current setting: it does not meet {}.** The female threshold is {:.2f} and the male threshold '
                       'is {:.2f}.'.format(failed_text, female_threshold, male_threshold))
 
    st.divider()
 
    # 6. Fairness metrics panel
 
    # Work out the four metrics from the current predictions and show each one with a pass or fail tag
    metrics = compute_metrics(y_true, predictions, group)
    st.subheader('Fairness Metrics')
    st.caption('Three of these are gaps between the two groups, so they read as fair when close to 0. The Disparate '
               'Impact Ratio compares the groups as a ratio, so it reads as fair when close to 1. Each turns green '
               'when it sits within the tolerance set in the control panel. The technical definition is on the '
               'question mark, and the note under each value gives the same measure as a count of patients.')
 
    # Each metric compares two rates, so it can be read back as a count of patients per hundred
    # A decimal on its own does not say how many people it covers, which is what a reader needs to judge it
    female_mask_now = group == 0
    male_mask_now = group == 1
    female_flagged = predictions[female_mask_now].mean() * 100
    male_flagged = predictions[male_mask_now].mean() * 100
    female_caught = predictions[female_mask_now & (y_true == 1)].mean() * 100
    male_caught = predictions[male_mask_now & (y_true == 1)].mean() * 100
    female_alarmed = predictions[female_mask_now & (y_true == 0)].mean() * 100
    male_alarmed = predictions[male_mask_now & (y_true == 0)].mean() * 100
    female_correct_flag = y_true[female_mask_now & (predictions == 1)].mean() * 100 if (predictions[female_mask_now] == 1).any() else 0
    male_correct_flag = y_true[male_mask_now & (predictions == 1)].mean() * 100 if (predictions[male_mask_now] == 1).any() else 0
 
    # The wording for each metric opens in bold with the question it asks, then answers it with the counts in bold
    plain_meaning = {
        'Demographic Parity Difference':
            '**Are both groups flagged equally often?** Out of every 100 female patients, **{:.0f}** are flagged as '
            'having disease. Out of every 100 male patients, **{:.0f}** are flagged as having disease.'.format(female_flagged, male_flagged),
        'Equalised Odds Difference':
            '**Does the model catch disease and raise false alarms equally for both groups?** Among patients who truly '
            'have disease, the model correctly flags **{:.0f}** as having disease in every 100 women and **{:.0f}** in '
            'every 100 men. Among patients who do not have disease, the model wrongly flags **{:.0f}** as having '
            'disease in every 100 women and **{:.0f}** in every 100 men. The score is the larger of these two gaps, '
            'the gap in disease caught and the gap in false alarms.'.format(female_caught, male_caught, female_alarmed, male_alarmed),
        'Predictive Parity Difference':
            '**Can a positive flag be trusted equally for both groups?** When a female patient is flagged as having '
            'disease, the flag is correct **{:.0f}** times in every 100. When a male patient is flagged as having '
            'disease, the flag is correct **{:.0f}** times in every 100.'.format(female_correct_flag, male_correct_flag),
        'Disparate Impact Ratio':
            '**Are the two groups flagged in similar proportion?** The group flagged as having disease less often is '
            'flagged at **{:.0f}** percent of the rate of the group flagged as having disease more often. A value of '
            '100 percent would mean the two groups are flagged as having disease equally often, and the further below '
            '100 it falls, the wider the gap between them.'.format((min(female_flagged, male_flagged) / max(female_flagged, male_flagged) * 100)
                          if max(female_flagged, male_flagged) > 0 else 0)}
 
    # Two metrics to a row rather than four, so each note has half the width instead of a quarter
    columns_row_one = st.columns(2)
    columns_row_two = st.columns(2)
    metric_rows = [columns_row_one, columns_row_one, columns_row_two, columns_row_two]
    for i, name in enumerate(metrics):
        value = metrics[name]
        column = metric_rows[i][i % 2]
        # Disparate Impact is a ratio that is fair near 1, while the others are differences that are fair near 0
        if name == 'Disparate Impact Ratio':
            within = value >= (1 - tolerance)
        else:
            within = value <= tolerance
        # The help text explains what each metric measures, shown as a help tooltip next to the value
        column.metric(name, '{:.4f}'.format(value), help=METRIC_HELP[name])
        if within:
            column.success('Within Tolerance')
        else:
            column.error('Outside Tolerance')
        # The reading in patients sits in a note under each value, since a decimal on its own does not say how many people it covers
        column.info(plain_meaning[name])
 
    # The first and last measures are built from the same two rates, so only three of the four carry separate information
    st.caption('Demographic Parity Difference and the Disparate Impact Ratio are two views of the same comparison. '
               'One subtracts the two flagging rates, the other divides them, so only three of the four measures are '
               'independent.')
 
    st.divider()
 
    # 7. Where the fairness metrics come from
 
    # Split recall, precision and the two error rates by sex, so under-diagnosis shows up as a higher female false negative rate
    performance_metrics = {'Recall': recall_score, 'Precision': safe_precision,
                           'False Negative Rate': false_negative_rate, 'False Positive Rate': false_positive_rate}
    performance_by_sex = MetricFrame(metrics=performance_metrics, y_true=y_true, y_pred=predictions,
                                     sensitive_features=group)
    by_group = performance_by_sex.by_group
 
    # Predictive parity and equalised odds are read straight off these values, so they belong under the metrics rather than beside them
    # Keeping them in a drill-down leaves the panel above uncluttered while the working stays available
    with st.expander('Where these numbers come from: performance by {}'.format(group_label.lower())):
        st.caption('Two of the measures above are read straight off the values below. Predictive Parity is the gap '
                   'between female and male precision, and Equalised Odds is the larger of the recall gap and the '
                   'false positive rate gap. Recall, also called the true positive rate, is the proportion of real '
                   'disease cases the model correctly flags. The false negative rate is the proportion of real disease '
                   'cases the model misses, and this is where under-diagnosis shows up, which matters most in a clinical '
                   'setting. The false positive rate is the proportion of healthy patients flagged in error.')
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
 
        # Point out what the gap in the two false negative rates means, since this is where under-diagnosis appears
        # The rates themselves are in the boxes above, so the note reads them rather than repeating the figures
        # A rate worked out over a small group is less reliable, so the note says how many patients the female rate rests on
        female_true_cases = int(((group == 0) & (y_true == 1)).sum())
        callout = ('Any gap between the two false negative rates is where under-diagnosis would surface, since it '
                   'means one group\'s real cases are missed more often than the other\'s.')
        if female_true_cases < 30:
            callout += (' The female rate rests on only {} patients who truly have disease, so it is best read as a '
                        'strong signal of direction rather than an exact measurement.'.format(female_true_cases))
            
        st.info(callout)
 

with tab_errors:
 
    # A short framing so the clinical language below reads as the cost the developer weighs, not as advice to a clinician
    st.caption('These panels show who the current model and thresholds affect. The developer reads them to weigh the '
               'clinical cost of the errors before deciding whether the model is ready to hand over to a clinician.')
 
    # 8. Outcomes for a hundred patients
 
    # A rate is easier to judge as a count of people than as a decimal, so each error rate is drawn as a hundred squares
    # The two rates have different denominators, since a missed case is counted against the patients who truly have disease and a false alarm against the healthy ones
    st.subheader('What Happens to 100 Patients')
    st.caption('Each square is one patient, under the model and thresholds currently selected. The top row takes 100 '
               'patients who truly have disease and shades the ones the model misses. The bottom row takes 100 healthy '
               'patients and shades the ones it flags in error. Reading the two groups side by side shows how the same '
               'decision lands differently on each.')
 
    female_disease = (group == 0) & (y_true == 1)
    male_disease = (group == 1) & (y_true == 1)
    female_healthy = (group == 0) & (y_true == 0)
    male_healthy = (group == 1) & (y_true == 0)
    female_missed_per_hundred = int(round((predictions[female_disease] == 0).mean() * 100)) if female_disease.sum() > 0 else 0
    male_missed_per_hundred = int(round((predictions[male_disease] == 0).mean() * 100)) if male_disease.sum() > 0 else 0
    female_alarm_per_hundred = int(round((predictions[female_healthy] == 1).mean() * 100)) if female_healthy.sum() > 0 else 0
    male_alarm_per_hundred = int(round((predictions[male_healthy] == 1).mean() * 100)) if male_healthy.sum() > 0 else 0
 
    # Missed cases sit on the top row against the patients who truly have disease, drawn in red over the ones the model caught
    missed_female_column, missed_male_column = st.columns(2)
    missed_female_column.markdown('**Of 100 female patients who truly have disease**')
    missed_female_column.markdown(icon_array(female_missed_per_hundred, '#7b241c', '#3cae6d'), unsafe_allow_html=True)
    missed_female_column.caption('{} are missed, {} are correctly flagged.'.format(female_missed_per_hundred,
                                                                                   100 - female_missed_per_hundred))
    missed_male_column.markdown('**Of 100 male patients who truly have disease**')
    missed_male_column.markdown(icon_array(male_missed_per_hundred, '#7b241c', '#3cae6d'), unsafe_allow_html=True)
    missed_male_column.caption('{} are missed, {} are correctly flagged.'.format(male_missed_per_hundred,
                                                                                 100 - male_missed_per_hundred))
 
    # False alarms sit on the bottom row against the healthy patients, drawn in orange over the ones the model left alone
    alarm_female_column, alarm_male_column = st.columns(2)
    alarm_female_column.markdown('**Of 100 healthy female patients**')
    alarm_female_column.markdown(icon_array(female_alarm_per_hundred, '#e67e22', '#e2e8f0'), unsafe_allow_html=True)
    alarm_female_column.caption('{} are flagged in error, {} are correctly cleared.'.format(female_alarm_per_hundred,
                                                                                            100 - female_alarm_per_hundred))
    alarm_male_column.markdown('**Of 100 healthy male patients**')
    alarm_male_column.markdown(icon_array(male_alarm_per_hundred, '#e67e22', '#e2e8f0'), unsafe_allow_html=True)
    alarm_male_column.caption('{} are flagged in error, {} are correctly cleared.'.format(male_alarm_per_hundred,
                                                                                          100 - male_alarm_per_hundred))
 
    st.divider()
 
    # 9. Missed patients (false negatives)
 
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
        st.caption('As with the missed patients above, there is no probability score to show for the threshold optimiser.')
        missed_table = pd.DataFrame({'Position': missed, group_label: missed_sex})
    else:
        missed_table = pd.DataFrame({'Position': missed, group_label: missed_sex, 'Predicted Probability': proba[missed]})
 
    # Let the user filter the table by sex, so the missed cases of one group can be looked at on their own
    missed_filter = st.selectbox('Filter Missed Patients by {}'.format(group_label), ['All', 'Female', 'Male'])
    if missed_filter != 'All':
        missed_table = missed_table[missed_table[group_label] == missed_filter]
    st.dataframe(missed_table.round(4), hide_index=True)
 
    st.divider()
 
    # 10. False alarms (false positives)
 
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
 

with panel_overview:
 
    # 11. Dataset overview
 
    # Give a plain description of the dataset in view, so the user knows what they are looking at before the analysis
    st.subheader('About This Dataset')
    st.caption('This tab describes the dataset currently selected in the control panel. It gives the size of the data, '
               'the meaning of every feature the model uses, and the disease rate in each group. Switching the dataset '
               'in the control panel updates everything shown here, so each dataset can be read on its own terms.')
 
    # Show the full size after preprocessing, the size of the test set the dashboard works on, and the number of features
    # The test-set count is read live from the loaded data, while the full count is held in a dict since the saved data holds only the test set
    n_test = len(probabilities)
    n_features = len(FEATURE_INFO[prefix])
    overview_total, overview_test, overview_features = st.columns(3)
    overview_total.metric('Total Patients', '{}'.format(DATASET_TOTAL[prefix]),
                          help='The number of patients left after cleaning the raw data, before the split into training and test sets.')
    overview_test.metric('Test Set (shown here)', '{}'.format(n_test),
                         help='The held-out portion the dashboard runs on. The fairness measures and error tables are all worked out on these patients. The one exception is the disease rate below, which is measured across the full dataset.')
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
 
    st.divider()
 
    # 12. Disease rate in the data
 
    # The gap between the two disease rates is the reason the fairness measures pull against each other, so it belongs with the description of the data
    st.subheader('Disease Rate by {} (Base Rate)'.format(group_label))
    st.caption('The proportion of female and male patients who actually have heart disease, known as the base rate. This is '
               'measured across the full dataset before the train and test split, so it describes the data as a whole '
               'rather than the test set the other panels work on. The wider this gap, the harder it becomes to satisfy '
               'the fairness measures at the same time, which is what the Fairness Metrics tab reports. A large gap on '
               'its own is not proof of bias, though. It might reflect a genuine difference in how often the disease '
               'occurs, or a history of under-diagnosis in one group, and the data alone cannot tell the two apart.')
    female_rate = base_rate['disease_rate'].values[0]
    male_rate = base_rate['disease_rate'].values[1]
    gap = male_rate - female_rate
    rate_female, rate_male, rate_gap = st.columns(3)
    rate_female.metric('Female', '{:.2f}%'.format(female_rate * 100))
    rate_male.metric('Male', '{:.2f}%'.format(male_rate * 100))
    rate_gap.metric('Gap', '{:.2f} points'.format(gap * 100),
                    help='The difference between the male and female disease rates, in percentage points.')


with panel_explain:
 
    # The SHAP explanations were worked out for the baseline model only, so this tab shows a note when a mitigation is chosen
    if mitigation_name != 'Baseline':
        st.info('The explanations on this tab are worked out for the baseline model only, so they appear once the '
                'baseline is selected in the control panel. The SHAP values were saved for the baseline model, since that '
                'is the one being audited. SMOTE-NC and reweighting retrain the model, so their explanations would need '
                'their own SHAP values, while the threshold optimiser leaves the model untouched and changes only the '
                'decision threshold. Select the baseline to view the explanations. To compare the mitigation methods, use '
                'the Fairness Metrics and Errors tabs, or Calibration & ROC and Dataset and Metric Comparison under '
                'Data and Analysis.')
    else:
 
        # 13. Gender-stratified SHAP panel
 
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
                    'male patients. That asymmetry describes how the model uses {0} rather than pointing to a clinical '
                    'cause. The model leans on {0} when assessing female patients but on the clinical measurements '
                    'when assessing male patients, and that is the pattern of under-diagnosis this project sets out '
                    'to investigate.'.format(group_label.lower(), male_sex_rank))
        else:
            st.info('Look at where {} sits in each of the two rankings. If it is a top feature for one group but not '
                    'the other, that asymmetry suggests the model is treating the two groups differently rather than '
                    'relying on the same clinical findings for both.'.format(group_label.lower()))

 
with panel_quality:
 
    # 14. Calibration by sex
 
    # Calibration is the fifth fairness metric, and the threshold optimiser has no probability, so the panel shows a note instead when it is chosen
    st.subheader('Calibration by {}'.format(group_label))
    st.caption('The fifth fairness metric asks whether a predicted risk score means the same thing for female and '
               'male patients. A model is well calibrated when, among the patients given a risk of around 0.7, about '
               '70 percent truly have disease, and this should hold for both groups. A curve on the diagonal means the '
               'scores can be trusted. A curve below it means the model overestimates the risk for that group, and a '
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
    st.caption('The curves show how well the model separates disease cases from healthy patients within each group, '
               'across every threshold. A curve closer to the top-left is better, and the AUC sums that up in one '
               'score running from 0.5, no better than chance, to 1, a perfect separation. When the two curves sit '
               'close together the model ranks both groups equally well. The fairness gap then comes from where the '
               'shared threshold falls rather than from the model itself, which points towards adjusting the '
               'threshold for each group. ROC is not one of the five fairness criteria. It is here to show where the '
               'Equalised Odds gap comes from.')
    if optimiser_chosen:
        st.info('An ROC curve ranks patients by their score, and the threshold optimiser does not produce one, so '
                'there is nothing to plot here either. The same two mitigations that restore the calibration curve '
                'above will restore this one.')
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
 
 
with panel_compare:
 
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
                      'Recall': 'A higher value is better, since this measures how many true disease cases the model correctly flags.'}
 
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
            # A difference metric is fair below the tolerance, while the ratio is fair above one minus the tolerance, with the four-fifths rule shown as a fixed reference
            if metric_choice in difference_metrics:
                ax.axhline(tolerance, linestyle='--', color='#999999', label='Tolerance ({:.2f})'.format(tolerance))
            elif metric_choice == 'Disparate Impact Ratio':
                ax.axhline(1 - tolerance, linestyle='--', color='#999999', label='Tolerance ({:.2f})'.format(1 - tolerance))
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
                ax.axhline(1 - tolerance, linestyle='--', color='#999999', label='Tolerance ({:.2f})'.format(1 - tolerance))
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
