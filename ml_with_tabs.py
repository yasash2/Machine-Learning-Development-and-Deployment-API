import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
import ydata_profiling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
def profiling(df):
    profile_report = df.profile_report()
    st_profile_report(profile_report)
def clean_data(df, choice):
    
        x1=df.columns
        for col in x1:
            try:
                df[col] = df[col].apply(pd.to_numeric)
            except:
                continue
        for char in x1:
            n=list(char.split())
            if (len(n) > 1):
                g="_".join(n)
                df.rename(columns={char: g}, inplace=True)

    #---------------------------------------------------------------------------------    
        
        st.write(""" ## step 1:- (deleting columns)""")
        col1=st.multiselect("select a number",df.columns, key = "1")
        df.drop(col1,axis=1,inplace=True)
        st.write(df)
        
    #------------------------------------------------------------------------------------------------------------

        st.write(""" ## step-2:- (IMPUTING)""")
        x1=df.columns
        for col in x1:
            if (df[col].dtype =='O'):
                imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
                df[col]= imputer.fit_transform(df[[col]])
            else:
                imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
                df[col]= imputer.fit_transform(df[[col]])
        st.write(df)
        
        #--------------------------------------------------------------------------------------------------------
        st.write("""## step - 3:- outlier removal:-""")  
        col=st.selectbox("select your target variable:",df.columns,key = "2")
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr=q3-q1
        upper_limit = q3+1.5*iqr
        lower_limit = q1-1.5*iqr
        def limit_imputer(value):
            if value<lower_limit:
                return lower_limit
            elif value>upper_limit:
                return upper_limit
            else:
                return value
        df[col] = df[col].apply(limit_imputer) 
        st.write(df)
        #-----------------------------------------------------------------------------------------------------------

#        st.write(""" ## step-4: dummy variables:-""")
#        x1=df.columns
#        for col in x1:
#            if (df[col].dtype=="O"):
#                dummies=pd.get_dummies(df[[col]])
#                df=pd.concat([df,dummies],axis='columns')
#                df=df.drop([col],axis='columns')
#            else:
#                continue
#        st.write(df)    


        st.write(""" ## step -4 Label Encoding""")
        def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
        df = Encoder(df)
        st.write(df)
        #---------------------------------------------------------------------------------------------------------------------
        

        st.write(""" ## step-5 :- feature scaling""")
        x1=df.columns
        p=choice
        scaler = StandardScaler()
        Y= df[p]  # target variable
        X = scaler.fit_transform(df.drop(columns=[p]))
        X = pd.DataFrame(data=X,columns=df.drop(columns=[p]).columns)
        df=pd.concat([Y,X],axis='columns')
        st.write(df)
        st.download_button("Download Cleaned Data", data = df.to_csv().encode('utf-8'),file_name='Cleaned_df.csv', mime= 'text/csv')
        #----------------------------------------------------------------------------------------------------
        split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        return split_size,X,Y
        
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
def classify(split_size,X,Y):

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
            #-----------------------------------------------------------------------------------------------------
            if st.button("start ploting hyper parameters"):

                classifier = RFC()
                classifier.fit(x_train, y_train)

                ## METRICS
                from sklearn.metrics import f1_score
                def calc_score(model, x1, y1, x2, y2):
                    model.fit(x1,y1)
                    predict = model.predict(x1)
                    f1 = f1_score(y1, predict,pos_label='positive',average='micro')
                    predict = model.predict(x2)
                    f2 = f1_score(y2, predict,pos_label='positive',average='micro')
                    return f1, f2

                def effect(train_score, test_score, x_axis, title):
                    plt.figure(figsize = (7,4), dpi = 120)
                    fig, ax = plt.subplots()
                    ax.plot(x_axis, train_score, color = 'red', label = 'train_Score')
                    ax.plot(x_axis, test_score, color = 'blue', label = 'test_Score')
                    plt.title(title)
                    plt.xlabel("parameter_value")
                    plt.ylabel("f1 score")
                    st.pyplot(fig)

                    d={"parameter_value": x_axis,
                    "f1-score":test_score}
                    dt=pd.DataFrame(d)
                    st.write(dt)
                    l=max(test_score)
                    st.write("maximum value is",l)
                    m=0
                    while(test_score[m]!=l):
                        m+=1
                    max_key=x_axis[m]
                    st.write("parameter value at max f1_score is : ",max_key)
                    return max_key    
                    
                    


                    
                
                ## FIRST PARAMETER (n_estimators)
                st.write(""" ## step-6:- setting up hyperparameters""")
                st.write("""### step-6.1:- N_estimators""")
                estimators = [i for i in range(1,400,10)]
                train = []
                test = []
                for i in estimators:  
                    model = RFC(class_weight = 'balanced_subsample',
                                n_estimators = i,
                                n_jobs = -1,
                                max_depth = 7,
                                random_state = 101)
                    f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
                    train.append(f1)
                    test.append(f2)
                n_est=effect( train, test, range(1,400,10) , 'n_estimators')


                ## SECOND PARAMETER(max samples )
                st.write("""### step-6.2:- max samples""")
                maxsamples = [i/1000 for i in range(1,400)]
                train = []
                test = []

                for i in maxsamples:  
                    model = RFC(class_weight = 'balanced_subsample', n_estimators = n_est,n_jobs = -1, max_depth = 7, random_state = 101, max_samples = i)
                    f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
                    train.append(f1)
                    test.append(f2)
                max_samp=effect( train, test, maxsamples , 'max samples')   
                

                ## THIRD PARAMETER (MAX FEATURES)
                st.write("""### step-6.3:- MAX FEATURES""")
                maxfeatures = range(1,X.shape[1])
                train = []
                test = []
                for i in maxfeatures:  
                    model = RFC(class_weight = 'balanced_subsample', n_estimators = n_est,max_samples=max_samp,n_jobs = -1, max_depth = 7, random_state = 101, max_features = i)
                    f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
                    train.append(f1)
                    test.append(f2)
                max_feat=effect( train, test, maxfeatures , 'number of max features for individual tree')    

                #n_est=st.number_input("enter n_estimators",min_value=0,max_value=600,value=0,step=5)
                #max_samp=st.number_input("enter max_samples",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
                #max_feat=st.number_input("enter max_features",min_value=0.0,max_value=100.0,value=0.0,step=0.5)
                

                clf=RFC(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                criterion='gini', max_depth=None, max_features=max_feat,
                                max_leaf_nodes=None, max_samples=max_samp,
                                min_impurity_decrease=0.0,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=n_est,
                                n_jobs=None, oob_score=False, random_state=None,
                                verbose=0, warm_start=False)
                clf.fit(x_train,y_train)
                st.write(clf.get_params())
                predicted_values = clf.predict(x_train)
                st.write(classification_report(y_train, predicted_values))
                predicted_values = clf.predict(x_test)
                st.write(classification_report(y_test, predicted_values))   
                st.download_button("Download Model", data=pickle.dumps(clf),file_name="model.pkl")


def regress(split_size,X,Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
        #-----------------------------------------------------------------------------------------------------
        if st.button("Start plotting Hyperparameters"):

            classifier = RFR()
            classifier.fit(x_train, y_train)

            ## METRICS
            from sklearn.metrics import f1_score
            def cal_score(model, x1, y1, x2, y2):
                model.fit(x1,y1)
                predict = model.predict(x1)
                f1 = r2_score(y1, predict)
                predict = model.predict(x2)
                f2 = r2_score(y2, predict)
                return f1, f2

            def effec(train_score, test_score, x_axis, title):
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.plot(x_axis, train_score, color = 'red', label = 'train_Score')
                ax.plot(x_axis, test_score, color = 'blue', label = 'test_Score')
                plt.title(title)
                plt.xlabel("parameter_value")
                plt.ylabel("f1 score")
                st.pyplot(fig)

                d={"parameter_value": x_axis,
                "f1-score":test_score}
                dt=pd.DataFrame(d)
                st.write(dt)
                l=max(test_score)
                st.write("maximum value is",l)
                m=0
                while(test_score[m]!=l):
                    m+=1
                max_key=x_axis[m]
                st.write("parameter value at max f1_score is : ",max_key)
                return max_key    
                
            ## FIRST PARAMETER (n_estimators)
            st.write(""" ## step-6:- setting up hyperparameters""")
            st.write("""### step-6.1:- N_estimators""")
            estimators = [i for i in range(1,400,10)]
            train = []
            test = []
            for i in estimators:  
                model = RFR(
                            n_estimators = i,
                            n_jobs = -1,
                            max_depth = 7,
                            random_state = 101)
                f1, f2 = cal_score(model, x_train, y_train, x_test, y_test)
                train.append(f1)
                test.append(f2)
            n_est=effec( train, test, range(1,400,10) , 'n_estimators')


            ## SECOND PARAMETER(max samples )
            st.write("""### step-6.2:- max samples""")
            maxsamples = [i/1000 for i in range(1,400)]
            train = []
            test = []

            for i in maxsamples:  
                model = RFR( n_estimators = n_est,n_jobs = -1, max_depth = 7, random_state = 101, max_samples = i)
                f1, f2 = cal_score(model, x_train, y_train, x_test, y_test)
                train.append(f1)
                test.append(f2)
            max_samp=effec( train, test, maxsamples , 'max samples')   
            

            ## THIRD PARAMETER (MAX FEATURES)
            st.write("""### step-6.3:- MAX FEATURES""")
            maxfeatures = range(1,X.shape[1])
            train = []
            test = []
            for i in maxfeatures:  
                model = RFR( n_estimators = n_est,max_samples=max_samp,n_jobs = -1, max_depth = 7, random_state = 101, max_features = i)
                f1, f2 = cal_score(model, x_train, y_train, x_test, y_test)
                train.append(f1)
                test.append(f2)
            max_feat=effec( train, test, maxfeatures , 'number of max features for individual tree')    

            #n_est=st.number_input("enter n_estimators",min_value=0,max_value=600,value=0,step=5)
            #max_samp=st.number_input("enter max_samples",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
            #max_feat=st.number_input("enter max_features",min_value=0.0,max_value=100.0,value=0.0,step=0.5)
            

            clf=RFR(bootstrap=True, ccp_alpha=0.0,
                            criterion='squared_error', max_depth=None, max_features=max_feat,
                            max_leaf_nodes=None, max_samples=max_samp,
                            min_impurity_decrease=0.0,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=n_est,
                            n_jobs=None, oob_score=False, random_state=None,
                            verbose=0, warm_start=False)
            clf.fit(x_train,y_train)
            st.write(clf.get_params())
            st.download_button("Download Model", data=pickle.dumps(clf),file_name="model.pkl")

def unsupervised(df):
    xun1=df.columns
    for col in xun1:
        try:
            df[col] = df[col].apply(pd.to_numeric)
        except:
            continue
    for char in xun1:
        n=list(char.split())
        if (len(n) > 1):
            g="_".join(n)
            df.rename(columns={char: g}, inplace=True)
    
#---------------------------------------------------------------------------------    
    st.write(""" ## step 1:- (deleting columns)""")
    colun1=st.multiselect("select a number",df.columns, key = "4")
    df.drop(colun1,axis=1,inplace=True)
    st.write(df)
    st.write(df.info())
#------------------------------------------------------------------------------------------------------------

    st.write(""" ## step-2:- (IMPUTING)""")
    xun2=df.columns
    for col in xun2:
        if (df[col].dtype =='O'):
            imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
            df[col]= imputer.fit_transform(df[[col]])
        else:
            imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
            df[col]= imputer.fit_transform(df[[col]])
    st.write(df)
    
    #--------------------------------------------------------------------------------------------------------


    #st.write(""" ## step-5 :- feature scaling""")
    #r=df.columns
    #scaler = MinMaxScaler()
    #df= scaler.fit_transform(df)
    #st.write(df)
    
    st.write(""" ## step-3 :- feature scaling""")
    r=df.columns
    p=st.multiselect("select your target variables:",r, key = "5")
    scaler = MinMaxScaler()
    if st.checkbox('perform feature scaling',value=True):
      Y= df[p]  # target variables)
      dY=scaler.fit_transform(Y)
      df= pd.DataFrame(data=dY,columns=Y.columns)
      st.write(df)
    else:
          Y= df[p]
          df= pd.DataFrame(data=Y,columns=Y.columns)
          st.write(df)
    y_list=(Y.keys())
    battu=st.button("start the k-means process")
    if "load_state" not in st.session_state:
        st.session_state.load_state=False
    if battu or st.session_state.load_state :
        st.session_state.load_state=True
    #----------------------------------------------------------------------------------------------------
        st.write("""# step-4:- Performing K-means algorithm """)
        process=st.radio("perform these processes",["finding k-value","k-means and plotting graphs"])
        if process=="finding k-value":
            sse = []
            colu=df.columns
            loc=list[col]
            st.write(loc)
            k_rng = range(1,10)
            for k in k_rng:
                km = KMeans(n_clusters=k)
                km.fit(df[colu])
                sse.append(km.inertia_)
            plt.figure(figsize = (7,4), dpi = 120)
            fig, ax = plt.subplots()
            ax.plot(k_rng, sse, color = 'red', label = 'train_Score')
            plt.title("finding k value")
            plt.xlabel("k_value")
            plt.ylabel("sum of squared error")
            st.pyplot(fig)
        elif process=="k-means and plotting graphs":  
            num_ip=st.number_input("select k-value",min_value=1,max_value=15,value=4,step=1)
            st.write(num_ip)
            colu=df.columns
            plt.figure(figsize = (7,4), dpi = 120)
            fig, ax = plt.subplots()
            ax.scatter(df[y_list[0]],df[y_list[1]],color='green')
            plt.title('scatter plot before k-means')
            plt.xlabel(y_list[0])
            plt.ylabel(y_list[1])
            plt.legend()
            st.pyplot(fig)
            km = KMeans(n_clusters=num_ip)
            y_predicted = km.fit_predict(df[colu])
            
            df['cluster']=y_predicted
            st.write(df)
            centers=km.cluster_centers_
            st.write(centers)
            if num_ip==3:
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                df3 = df[df.cluster==2]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(df3[y_list[0]],df3[y_list[1]],color='red')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig) 
            elif num_ip==2:
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig) 
            elif num_ip==4:
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                df3 = df[df.cluster==2]
                df4 = df[df.cluster==3]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(df3[y_list[0]],df3[y_list[1]],color='red')
                ax.scatter(df4[y_list[0]],df4[y_list[1]],color='black')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig)   
            elif num_ip==5: 
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                df3 = df[df.cluster==2]
                df4 = df[df.cluster==3]
                df5 = df[df.cluster==4]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(df3[y_list[0]],df3[y_list[1]],color='red')
                ax.scatter(df4[y_list[0]],df4[y_list[1]],color='black')
                ax.scatter(df5[y_list[0]],df5[y_list[1]],color='yellow')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig)      
            elif num_ip==6: 
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                df3 = df[df.cluster==2]
                df4 = df[df.cluster==3]
                df5 = df[df.cluster==4]
                df6 = df[df.cluster==5]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(df3[y_list[0]],df3[y_list[1]],color='red')
                ax.scatter(df4[y_list[0]],df4[y_list[1]],color='black')
                ax.scatter(df5[y_list[0]],df5[y_list[1]],color='yellow')
                ax.scatter(df6[y_list[0]],df6[y_list[1]],color='magenta')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig)   
            elif num_ip==1:
                df1 = df[df.cluster==0]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig)  
            elif num_ip==7: 
                df1 = df[df.cluster==0]
                df2 = df[df.cluster==1]
                df3 = df[df.cluster==2]
                df4 = df[df.cluster==3]
                df5 = df[df.cluster==4]
                df6 = df[df.cluster==5]
                df7 = df[df.cluster==6]
                plt.figure(figsize = (7,4), dpi = 120)
                fig, ax = plt.subplots()
                ax.scatter(df1[y_list[0]],df1[y_list[1]],color='green')
                ax.scatter(df2[y_list[0]],df2[y_list[1]],color='blue')
                ax.scatter(df3[y_list[0]],df3[y_list[1]],color='red')
                ax.scatter(df4[y_list[0]],df4[y_list[1]],color='black')
                ax.scatter(df5[y_list[0]],df5[y_list[1]],color='yellow')
                ax.scatter(df6[y_list[0]],df6[y_list[1]],color='magenta')
                ax.scatter(df7[y_list[0]],df7[y_list[1]],color='cyan')
                ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
                plt.title('k-means plot')
                plt.xlabel(y_list[0])
                plt.ylabel(y_list[1])
                plt.legend()
                st.pyplot(fig)                    
                
def prediction(df, classifier):
    df = df.drop([str(choice)],axis = 1)
    column_list = df.columns
    lister = []
    
    for i in range(len(df.columns)):
        mini = df[column_list[i]].min()
        maxi = df[column_list[i]].max()
        if type(mini) == int or type(mini) == float:
            globals()['string%s' % i] = st.slider(column_list[i], float(mini), float(maxi))  
        else:
            globals()['string%s' % i] = st.selectbox(column_list[i], df[column_list[i]].unique())
            
    #st.write(df[column_list[1]])
    #st.write(len(df[column_list[1]].unique())) 
    uniquevalue_list = []
    Dict = {}
    for i in range(0,len(df.columns)):
        try:
            x = (globals()['string%s' % i])
            x = int(x)
        except ValueError:
            if len(df[column_list[i]].unique()) >= 2:
                Dict[i] = df[column_list[i]].unique()
                uniquevalue_list.append(df[column_list[i]].unique())
             
    for i in range(0, len(uniquevalue_list)):
        st.write(uniquevalue_list[i])
    st.write(Dict.values())


    le = LabelEncoder()
    
    for i in range(0,len(df.columns)):
        lister.append(globals()['string%s' % i ])
        if type(lister[i]) == float or type(lister[i]) == int:
            pass
        else:
            globals()['string%s' % i] = le.fit_transform(df[column_list[i]].unique())
            #st.write(print(globals()['string%s' % i]))

    ####
#    def Encoder(df):
#        columnsToEncode = list(df.select_dtypes(include=['category','object']))
#        le = LabelEncoder()
#        for feature in columnsToEncode:
#            try:
#                df[feature] = le.fit_transform(df[feature])
##            except:
 #               print('Error encoding '+feature)
  #      return df
    #df = Encoder(df)
    #st.write(df)
        #---------------------------------------------------------------------------------------------------------------------
        

    st.write(""" ## step-5 :- feature scaling""")
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    X = pd.DataFrame(data=X,columns=df.columns)
    df=pd.concat([X],axis='columns')
    st.write(df)
    prediction = classifier.predict([lister[i] for i in range(0,len(df.columns) )])
    print(prediction)
    return prediction
    
            

def model_deployment():
    if os.path.exists("model.pkl"):
        pickle_in = open('model.pkl', 'rb')
        classifier = pickle.load(pickle_in)
    pred = prediction(df, classifier)
    st.write(pred)
    





#=======================================================================#
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    


#========================================================================#
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    choice = st.sidebar.selectbox("Select the Target Variable",df.columns)
    tab_choice = st.sidebar.selectbox("Select Tab",["data profiling","supervised learning","unsupervised learning", "Deployment Tab"] )
    if tab_choice == "data profiling":
        profile_report = df.profile_report()
        st_profile_report(profile_report)
    elif tab_choice == "supervised learning":
        split_size,X,Y=clean_data(df, choice)
        rad=st.radio("choose your processing model",['regression','classification'])
        if (rad=="regression"):
             regress(split_size,X,Y)
        elif(rad=="classification"):
             classify(split_size,X,Y)
    elif tab_choice == "unsupervised learning":
        unsupervised(df)
    else:
        model_deployment()

#    profiling_tab,tab3,tab4,deployment_tab=st.tabs(["data profiling","supervised learning","unsupervised learning", "Deployment Tab"])
#
#    with profiling_tab:
#        profile_report = df.profile_report()
#        st_profile_report(profile_report)
#        
#    with tab3: 
#        split_size,X,Y=clean_data(df, choice)
##        rad=st.radio("choose your processing model",['regression','classification'])
#        if (rad=="regression"):
#             regress(split_size,X,Y)
#        elif(rad=="classification"):
#             classify(split_size,X,Y)        
#    with tab4:
#        unsupervised(df)
##    with deployment_tab:
#       model_deployment()


            

    
      
   
        





   

