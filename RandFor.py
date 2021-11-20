import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sn
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pylab as plt
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

dataset = pd.read_csv('D:\Scripts\peg\Dp_2019\Dp_04_2019_points1234.csv',  delimiter=';')
display(dataset.info())
print(dataset.describe())
corr = dataset.corr()
print(corr)
pca = PCA(n_components=5)
Z = dataset.drop('Soil5', axis=1, errors='ignore')

y = dataset['Soil5']




Xtrn, Xtest, ytrn, ytest = train_test_split(Z, y, test_size=0.25)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(Xtrn)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_scaled = ss.transform(Xtest)
X_test_pca = pca.transform(X_test_scaled)
y_train = np.array(ytrn)
y_test = np.array(ytest)
wavelength = Xtest['X']

lasso = LassoCV(tol=0.001).fit(X_train_pca, y_train)
feats = {}
for feature, importance in zip(dataset.columns, np.abs(lasso.coef_)):
    feats[feature] = importance
importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importance = importance.sort_values(by='Importance', ascending=False)
importance = importance.reset_index()
importance = importance.rename(columns={'index': 'Features'})

print("Lasso")
display(importance)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=1, max_depth=6, random_state=0, loss='ls')
gbr.fit(X_train_pca, y_train)
feats = {}
for feature, importance in zip(dataset.columns, gbr.feature_importances_):
    feats[feature] = importance
importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importance = importance.sort_values(by='Importance', ascending=False)
importance = importance.reset_index()
importance = importance.rename(columns={'index': 'Features'})

print("GBR")
display(importance)

rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=6, max_features='auto')
rf.fit(X_train_pca, y_train)
feats = {}
for feature, importance in zip(dataset.columns, rf.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})

regressor = xgb.XGBRegressor(n_estimators=100, reg_lambda=1,gamma=0, max_depth=6)
regressor.fit(X_train_pca,y_train)
feats = {}
for feature, importance in zip(dataset.columns, regressor.feature_importances_):
    feats[feature] = importance
importances1 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances1 = importances1.sort_values(by='Importance', ascending=False)
importances1 = importances1.reset_index()
importances1 = importances1.rename(columns={'index': 'Features'})


print("RandomForest")
display(importances)
print("XGBoost")
display(importances1)
#importances11 = np.array(importances1)

#plot_importance(regressor, max_num_features=10)
#sorted_idx = regressor.feature_importances_.argsort()
#plt.barh(Z.columns[sorted_idx][:10], regressor.feature_importances_[sorted_idx][:10])
#plt.xlabel("Xgboost Feature Importance")
#fig1.savefig('xgboost.png', figsize=(50, 40), dpi=1000)
#plt.show()

predict = lasso.predict(X_test_pca)
predictions = rf.predict(X_test_pca)
prediction = regressor.predict(X_test_pca)
prediction1 = gbr.predict(X_test_pca)
# Calculate the absolute errors
error = abs(predict - y_test)
errors = abs(predictions - y_test)
errors1 = abs(prediction - y_test)
errors2 = abs(prediction1 - y_test)
#print(error)
#print(errors)
#print(errors1)

r0 = r2_score(predict, y_test)
r = r2_score(predictions, y_test)
r1 = r2_score(prediction, y_test)
RMSE0 = mean_squared_error(predict, y_test)
RMSE = mean_squared_error(predictions, y_test)
RMSE1 = mean_squared_error(prediction, y_test)
r2 = r2_score(prediction1, y_test)
RMSE2 = mean_squared_error(prediction1, y_test)

print("Lasso")
print('r2 =',r0)
print('RMSE =',RMSE0)
print("RandomForest")
print('r2 =',r)
print('RMSE =',RMSE)
print("XGBoost")
print('r2 =',r1)
print('RMSE =',RMSE1)
print('r2 =',r2)
print('RMSE =',RMSE2)

score = cross_val_score(lasso, X_train_pca, y_train, cv=5)
scores = cross_val_score(rf, X_train_pca, y_train, cv=5)
scores1 = cross_val_score(regressor, X_train_pca, y_train, cv=5)
scores2 = cross_val_score(gbr, X_train_pca, y_train, cv=5)
print("5ти кратная перекрестная проверка r2")
print("Lasso")
print(score)
print("RandomForest")
print(scores)
print("XGBoost")
print(scores1)
print("GBR")
print(scores2)

#data = dict(col1=wavelength, col2=error, col3=errors, col4=errors1)
#df = pd.DataFrame(data)
#df.to_csv(r'd:/Scripts/peg/Dp_2019/out_Dp_06_2019_points234.csv', sep=';', index=False)

#data = dict(col1=wavelength, col2=1, col3=2, col4=3, col5=4, col6=5,col7=6,col8=7,col9=8,col10=9,col11=10,col12=11,col13=12,col14=13,col15=14,col16=15)
df = pd.DataFrame(corr)
sn.heatmap(corr, annot=True)
plt.show()
df.to_csv(r'd:/Scripts/peg/Dp_2019/out_Dp_04_2019_points1234.csv',sep=';', index=False)

print("Lasso")
print("Правильность на обучающем наборе: {:.3f}".format(lasso.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(lasso.score(X_test_pca, y_test)))

print("RandomForest")
print("Правильность на обучающем наборе: {:.3f}".format(rf.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(rf.score(X_test_pca, y_test)))

print("XGBoost")
print("Правильность на обучающем наборе: {:.3f}".format(regressor.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(regressor.score(X_test_pca, y_test)))

print("GBR")
print("Правильность на обучающем наборе: {:.3f}".format(gbr.score(X_train_pca, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(gbr.score(X_test_pca, y_test)))
