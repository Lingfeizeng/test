import yaml
import pandas as pd
import match_n_model
import numpy as np


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 100)
pd.options.display.float_format = '{:.12f}'.format



fluoride_40C = pd.read_pickle('all_fluoride_40C.pkl')
fluoride_40C.dtypes

# fluoride_40C.dtypes()
# pd.DataFrame.to_csv(fluoride_40C, "fluoride_40C.csv", float_format='%f')
# f = pd.read_csv('fluoride_40C.csv')
# f.head(5)


#d = match_n_model.Match_n_model.load_dict('database.yaml')
# list(d)
# ['cas', 'chemical', 'date', 'function', 'group', 'inci', 'ingredient', 'phase', 'special']

# phase_comp =d.get('phase')
# pd.DataFrame.to_csv(phase_comp, "phase_comp.csv")
# # list(phase_comp)
# # phase_comp.head(5)
# cas = d.get('cas')
# pd.DataFrame.to_csv(cas, "cas.csv")
# # list(cas)
# # cas.head(5)
# # list(phase_comp)
# chemical = d.get('chemical')
# pd.DataFrame.to_csv(chemical, "chemical.csv")
# # chemical.head(5)
# ingredient = d.get('ingredient')
# pd.DataFrame.to_csv(ingredient, "ingredient.csv")
# # ingredient.head(5)
# # date = d.get('date')
# function = d.get('function')
# pd.DataFrame.to_csv(function, "function.csv")
# # function.head(5)
# group = d.get('group')
# pd.DataFrame.to_csv(group, "group.csv")
# # group.head(5)
# inci = d.get('inci')
# pd.DataFrame.to_csv(inci, "inci.csv")
# # inci.head(5)
# special = d.get('special')
# pd.DataFrame.to_csv(special, "special.csv")
# # special.head(5)





# np.shape(inci) # (30742, 325)
# np.shape(ingredient) # (30848, 2005)
# np.shape(chemical) # (30742, 1691)
# np.shape(cas)
# np.shape(group)
# np.shape(function)
# np.shape(special) # (30848, 636)
phase_comp = pd.read_csv('phase_comp.csv', float_precision='round_trip')
phase_comp['recipe'] = phase_comp['recipe'].astype(str)
phase_comp.dtypes


X = match_n_model.Match_n_model(comp_type='phase', comp_data=phase_comp, test_type='fluoride', test_data=fluoride_40C, conditions='40C')

X.fit(linear=True, guess_initial=True, exponential=True, overwrite=True, plot=True)

# matched dataframe with fitting details
matched_raw = X.matched_raw
np.shape(matched_raw) # (4630, 145)
# matched dataframe for ML
matched = X.matched_values
matched.head(5)
np.shape(matched)  # (4630, 126)

#matched dataframe for ML where values have been converted to %change
pct_change = X.matched_changes # 4630 rows x 126 columns


X.model(pred_type='values', cv=10, selection=False, scaling=True, test_size=0.2, model_name='GradientBoostingRegressor', CI=80, sequential=False, ignore=[], export=True)

d_model = match_n_model.Match_n_model.load_dict('phase_fluoride_40C_80CI.yaml')
print(d_model)