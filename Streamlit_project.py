import pandas as pd
import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, jaccard_score, f1_score

# st.title('Customer Segmentation')

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {850}px;
        padding-top: {0.5}rem;
        padding-right: {0.5}rem;
        padding-left: {0.5}rem;
        padding-bottom: {0.5}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
st.markdown("<h1 style='text-align: center; color: Black;'>Customer Segmentation</h1>", unsafe_allow_html=True)

st.write("""
# Dataset
""")

# dataset_name = st.sidebar.selectbox(
#     'Select Dataset',
#     ('Iris', 'Breast Cancer', 'Wine')
# )
# dataset_name = Customer_Segmentation
# st.write(f"## {Customer_Segmentation} Dataset")
#

data_viz = pd.read_csv(r'C:\Users\Lenovo\Downloads\data_viz1.csv')


X= pd.read_csv(r'C:\Users\Lenovo\Downloads\customer_train_features.csv')
y= pd.read_csv(r'C:\Users\Lenovo\Downloads\customer_train_out.csv')


st.dataframe(data_viz.head(10))
st.write('Shape of dataset:', X.shape)
st.write('Number of classes in dataset:', len(np.unique(y)))
columns = np.array(data_viz.columns)
# st.write('Columns in dataset:' ,columns.tolist() )



#### Data Visualization ####
st.write("""
# Data Visualization
""")
######
st.write(""" ### Segmentation variation with Age & Graduation or Marriage """)
selected_class = st.radio("Select Class", ['Graduated','Ever_Married'])
st.write("Selected Class:", selected_class)
st.write("Selected Class Type:", type(selected_class))

import seaborn as sns
fig1 = plt.figure(figsize=(8,6))
sns.violinplot(x='Segmentation',y="Age",data=data_viz, hue=selected_class, split='True', palette='Set1')
st.pyplot(fig1)

st.write(""" ### Segmentation variation with Profession & Spending Score""")
selected_class1 = st.radio("Select Class", ['Profession','Spending_Score'])
st.write("Selected Class:", selected_class1)
st.write("Selected Class Type:", type(selected_class1))

fig2 = plt.figure()
df = data_viz[[selected_class1,"Segmentation"]].groupby([selected_class1, "Segmentation"]).size().unstack(level=0)
print(data_viz[selected_class1].unique())
for profession in data_viz[selected_class1].unique():
    df[profession] = df[profession]/sum(df[profession])
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df,cmap='coolwarm',annot=True,fmt='.2%')
st.pyplot(fig2)

#######
st.write(""" ### Segmentation variation with Work Experience & Family Size""")
selected_class2 = st.radio("Select Class", ('Work_Experience','Family_Size'))
st.write("Selected Class:", selected_class2)
st.write("Selected Class Type:", type(selected_class2))

group1 = pd.Series(data_viz.groupby((['Segmentation',selected_class2])).ID.agg('count'))
group1 = group1.to_frame()
stacked_bar = pd.pivot_table(group1,index=selected_class2,columns='Segmentation')
stacked_bar.reset_index(inplace=True)
stacked_bar.drop([selected_class2],axis=1,inplace=True)
if selected_class2 == 'Family_Size':
    for a in range(9):
        stacked_bar.iloc[a] = stacked_bar.iloc[a] / sum(stacked_bar.iloc[a])
if selected_class2 == 'Work_Experience' :
    for a in range(15):
        stacked_bar.iloc[a] = stacked_bar.iloc[a] / sum(stacked_bar.iloc[a])

stacked_bar = stacked_bar*100

# st.dataframe(stacked_bar)

x = np.array(stacked_bar.index)
y1 = np.array(stacked_bar[('ID', 'A')])
y2 = np.array(stacked_bar[('ID', 'B')])
y3 = np.array(stacked_bar[('ID', 'C')])
y4 = np.array(stacked_bar[('ID', 'D')])

######
fig3 = plt.figure(figsize=(8,6))
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1 + y2, color='y')
plt.bar(x, y4, bottom=y1 + y2 + y3, color='g')
if selected_class2 == 'Family_Size':
    plt.xlabel('Family Size')
    plt.xticks([i for i in range(1,10,1)])
else:
    plt.xlabel("Work Experience")
    plt.xticks([i for i in range(0, 15, 1)])
plt.ylabel("")
plt.legend(["A", "B", "C", "D"])
plt.title("")
plt.show()

st.pyplot(fig3)


#####
st.write("""### Segmentation vs Spending score""")
fig5 = plt.figure()
sns.countplot(data = data_viz,x='Spending_Score',hue = 'Segmentation',palette='inferno')
st.pyplot(fig5)


#### Model Predictions ####
X1 = pd.read_csv(r'C:\Users\Lenovo\Downloads\data_full.csv')

st.sidebar.write(""" # Model Predictions""")
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression','KNN', 'SVM', 'Random Forest')
)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    else :
        clf = LogisticRegression()
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=50)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = f1_score(y_test, y_pred,average = 'weighted')

st.sidebar.write(f'Classifier = {classifier_name}')
st.sidebar.write(f'Accuracy =', acc)

#### PLOT DATASET ####
st.sidebar.write(""" #### Projecting the data onto the 2 primary principal components""")
pca = PCA(2)
X_projected = pca.fit_transform(X_test)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

y_test = y_test.replace(['A','B','C','D'],[1,2,3,4])
y_test = y_test.values
fig4 = plt.figure()
plt.scatter(x1, x2,
        c=y_test, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.sidebar.pyplot(fig4)

