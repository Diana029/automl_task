from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
api = Api(app)

#номера фолдов
fold_nums = [0, 2, 4, 3, 1]
#количество фолдов
cnt_folds = 5 

#numerical features names
numerical_feat = ['col_5','col_9','col_12','col_22','col_25','col_37','col_55','col_56','col_59','col_60','col_86','col_95', 'col_119','col_125']

#all_features names
all_feat = ['col_'+str(i) for i in range(130)] + ['row_id']

#categorical features names
cat_cols = list(set(all_feat)-set(numerical_feat)-set(['row_id']))

# argument parsing
parser = reqparse.RequestParser()
for elem in all_feat:
    parser.add_argument(elem)

#load train folds data for catagorical encoding
train_folds = pd.read_pickle("/Users/dee/sber_automl_task/train_folds.pkl")

class model_lgb_prediction(Resource):
    def post(self):
        args = parser.parse_args()
        test_df = pd.DataFrame(columns=all_feat)
        for elem in all_feat:
            test_df.loc[0,elem] = args[elem]
 
        #датафрейм теста
        test_new = pd.DataFrame(columns = numerical_feat+['row_id']+[feat_name+'_mean_encoded' for feat_name in cat_cols])
        #preprocess numerical features in test
        for col in numerical_feat:
            test_df[col] = float(test_df[col])
               
        #preprocess categorical features in test
        for i in range(0,cnt_folds):
            for col in cat_cols:
                means = test_df[col].map(train_folds.groupby(col).target.mean())
                test_df[col+'_mean_encoded'] = means
                test_new = test_new.append(test_df)
        
        test_new.drop(cat_cols, axis=1, inplace=True)
        if test_new.isnull().values.any():
            prior = train_folds['target'].mean()      
            test_new.fillna(prior, inplace=True)
        #devide into x & ids
        X_test = test_new.drop(['row_id'], axis=1)
        ids_test = test_new[['row_id']]
        prediction = np.zeros(ids_test.shape[0])
        #forecast
        for i in range(0,cnt_folds):
            model_lgb = pickle.load(open("/Users/dee/sber_automl_task/model_lgb_"+str(i)+".pkl", 'rb'))
            y_pred_test = model_lgb.predict(X_test, num_iteration = model_lgb.best_iteration)
            prediction += y_pred_test
        prediction /= cnt_folds    
        test_sub = pd.concat([ids_test.reset_index(), pd.DataFrame({'prediction': prediction})],axis=1)
        test_sub.drop(['index'], axis=1, inplace=True)
        #round predictions
        test_sub['prediction'] =[round(elem, 2) for elem in test_sub['prediction']]
        output = {'row_id': int(test_sub['row_id'][0]), 'prediction' : float(test_sub['prediction'][0])} 
        return output



api.add_resource(model_lgb_prediction, '/')


if __name__ == '__main__':
    app.run(debug=True)

