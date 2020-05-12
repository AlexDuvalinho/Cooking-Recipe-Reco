import pandas as pd

#%%
def GetRecipesInformationsbyId(id, pp_recipes, raw_recipes):
    """Return for given id recipes its name and its ingredients
        id :  
            list<int>, the ids of the recipes (i)
        pp_recipes: 
            pandas array, the df PP_recipes
        raw_recipes:
            pandas array, the df RAW_recipes
    :Returns:
         A pandas Dataframe
    :Example:
        pp_recipes = pd.read_csv('../../data/PP_recipes.csv')
        raw_recipes = pd.read_csv('../../data/RAW_recipes.csv')
        GetRecipeInformationsbyId([7], pp_recipes, raw_recipes)
    """
    records = raw_recipes[raw_recipes['id'].isin(pp_recipes[pp_recipes['i'].isin(id)]['id'].values)]
    return records[['name', 'ingredients']]

#%%

