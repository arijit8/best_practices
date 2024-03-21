# Check p-value for dependency with the target variable
#Univaraite analysis

#After label encoding
from scipy.stats import chi2_contingency
test = chi2_contingency(pd.crosstab(train['popularity'], train['Category_2']))
print(f'From Chi-squared contegency test we can clearly reject Null hypothesis at 5% significance level having a p-value {test[1]}. Thus Category_2 is independent of the popularity class and we can keep the feature     ')

# Helper function -- FEATURE SELECTION
# Feature importance
X_train, X_test, y_train, y_test = train_test_split(train.drop(['popularity'], axis= 1).values, train['popularity'].values, test_size=0.33, random_state=42, stratify=train['popularity'].values)

feature_clf = RandomForestClassifier(n_estimators= 1000,
                                    n_jobs= -1,
                                    random_state = 1)
feature_clf.fit(X_train, y_train)
ypred = feature_clf.predict(X_test)

importances = feature_clf.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
col_names = train.drop(['popularity'], axis= 1).columns
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.show()

# Create folds
if __name__ == '__main__':
    df = pd.read_csv('MPP_Dataset/Train.csv')
    
    # create a fold column and fill dummy value
    df['kfold'] = -1
    
    #Suffle the data
    df.sample(frac=1).reset_index(drop =True)
    
    #Create Kfolds
    kf = StratifiedKFold(n_splits= 5)
    
    # Fetch Lebels
    y = df.popularity.values
    
    
    for f, (t_, v_) in enumerate(kf.split(X = df, y = y)):
        '''
        t_ is the indicies of the data
        f is number of fold
        
        '''
        df.loc[v_, 'kfold'] = f
    df.to_csv('MPP_Dataset/train_folds.csv')




def run(folds):
    metric_score =[]
    df = pd.read_csv('MPP_Dataset/train_folds.csv')
    
    # Feature Columns
    features = [names for names in df.columns if names not in ('kfold', 'popularity')]
    num_cols = []
    
    # If NaN exists in catagorical cols
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
            
    # If non-tree based algo Labelencoding will do the job Else Onehotencoding
    lbl = LabelEncoder()
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = lbl.fit_transform(df[col])
    
    # Onehotencoding
    ohc = OneHotEncoder()
    for col in features:
    if col not in num_cols:
        df.loc[:, col] = ohc.fit_transform(df[col])
        
    
    #Get training data using folds
    df_train = df[df.kfold != folds].reset_index(drop = True)
    df_valid = df[df.kfold == folds].reset_index(drop = True)
    
    #Get training data
    X_train = df_train[features].values
    X_valid = df_valid[features].values
    
    #Train and valid labels
    y_train = df_train.popularity
    y_valid = df_valid.popularity
    
    cv_clf = RandomForestClassifier(n_estimators= 1000,
                                    n_jobs= -1,
                                    random_state = 1)
    cv_clf.fit(X_train, y_train)
    ypred_proba = cv_clf.predict_proba(X_valid)
    
    metric_score.append(log_loss(y_valid, ypred_proba))
    print(f'Score in fold {folds} is {log_loss(y_valid, ypred_proba)}')
    print("*"*20)
    print(f'Score in fold {folds} is {roc_auc_score(y_valid, ypred_proba, multi_class=='ovo')}')
    print('='*20)
    
if __name__ == '__main__':
    for k in range(5):
        run(k)
