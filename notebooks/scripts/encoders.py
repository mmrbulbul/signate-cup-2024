import numpy as np
import pandas as pd
# import umap

from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, categories='auto', k=1, f=1, 
                 noise_level=0, random_state=None):
        if type(categories)==str and categories!='auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state
        
    def add_noise(self, series, noise_level):
        return series * (1 + noise_level *   
                         np.random.randn(len(series)))
        
    def fit(self, X, y=None):
        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target']
                       .agg(['mean', 'count']))
            # Compute smoothing 
            smoothing = (1 / (1 + np.exp(-(avg['count'] - self.k) /                 
                         self.f)))
            # The bigger the count the less full_avg is accounted
            self.encodings[variable] = dict(self.prior * (1 -  
                             smoothing) + avg['mean'] * smoothing)
            
        return self
    
    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable], 
                                 inplace=True)
            unknown_value = {value:self.prior for value in 
                             X[variable].unique() 
                             if value not in 
                             self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                Xt[variable].replace(unknown_value, inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable], 
                                              self.noise_level)
        return Xt
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)




class LowDimFeature(BaseEstimator, TransformerMixin):
    """
    Create features based on dimesionality reduction techniques, eg: umap, tsne
    """

    def __init__(self, n_components=2,
                 is_umap=True, is_tsne=True,
                 tsne_perplexity=10):
        """_summary_

        Args:
            n_components (int, optional): no of componests. Defaults to 2.
            is_umap (bool, optional): If True use umap for dimentionality reduction. Defaults to True.
            is_tsne (bool, optional):If True use umap for dimentionality reduction. Defaults to True.
            tsne_perplexity (int, optional):  Defaults to 10.
        """
        self.is_umap = is_umap
        self.is_tsne = is_tsne
        self.n_components = n_components
        # self.reducer1 = umap.UMAP(n_components=n_components)
        self.reducer2 = TSNE(n_components=n_components,
                             init='random', perplexity=tsne_perplexity)

    def fit(self, X, y=None):
        if self.is_umap:
            self.reducer1.fit(X)
        if self.is_tsne:
            self.reducer2.fit(X)
        return self

    def transform(self, X, y=None):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        if self.is_umap:
            df1 = self.reducer1.transform(X)
            df1 = pd.DataFrame(df1, columns=[
                               f"umap{i+1}" for i in range(self.n_components)])
        if self.is_tsne:
            df2 = self.reducer2.transform(X)
            df2 = pd.DataFrame(df2, columns=[
                               f"tsne{i+1}" for i in range(self.n_components)])

        df = pd.concat([df1, df2], axis=1)
        return df