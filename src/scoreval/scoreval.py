import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from itertools import product
class DummyModel() :
    def __init__(self, name='') :
        self.name = name
        
class ScoreEval() :
    '''
    Main entrence for score evaluation. Create an instance of ScoreEval and run methods step by step. 
    '''
    def __init__(self, models, data=None, weight_cols=[]):
        self.scores = []
        self.models = models
        self.data = data
        self.metrics = []
        self.cutset = []
        self.ivset = []
        self.weight_cols = weight_cols
        # self.cmap = ['red', 'green', 'orange', 'blue', 'purple', 'pink', 'grey'] * 5
        self.cmap = [ 
                '#990066', 
                '#663333',
                '#FFCC00',
                '#33AA33',
                '#006699',
                '#FF7733',
                '#009999',
                '#CCCC00',
                '#663399',
                '#aa2211',
                '#003399',
                '#99CC00',
                '#663300',
                '#FFCCCC',
                '#666699',
            ] * 3
        
    def _initialize_plot(self, figsize=(16,8)) :
        
        fig, axes = plt.subplots(len(self.models),1, figsize=figsize)

        if type(axes) != np.ndarray : 
            return fig, np.array([axes])
        return fig, axes
    
    def _save(self, fig, plt, save_path=None) :
        if save_path :
            fig.savefig(save_path)
        else: 
            plt.show()
        plt.close()

    def run_score(self, X, Y, func=None) :
        '''
        X: The input data for your model. 
        Y: The input label of your data set. 
        func: customized predict function, take every model in self.models and X as input. 
        '''
        self.scores = []
        for model in self.models :
            if func :
                pred = func(model, X)
                # pred = pred.reshape((pred.shape[0], 1))
            else :
                pred = model.predict(X)
            pred = pd.DataFrame.from_records(pred)
            pred.columns=['score']
            pred['label'] = Y
            # final = Y.merge(pred, how='inner', left_index=True, right_index=True)
            if self.weight_cols :
                pred[self.weight_cols] = self.data[self.weight_cols]
            else :
                pred[['weight1', 'weight2']] = 1
            self.scores.append(pred)

    def run_score_breakdown(self, break_col=[], break_val=[], filter_col=[], fval=[], fops=lambda x,y: x==y, id_col='id', score_col='score', label_col='label', weight_cols=[None, None], score_scale=0.01) :
            scores = []
            cols = [break_col, filter_col, id_col, score_col, label_col]
            w1, w2 = weight_cols
            if w1 :
                cols.append(w1)
            if w2 :
                cols.append(w2)
            # Select columns
            data = self.data[cols].rename(
                columns={
                    id_col: 'id', score_col: 'score', label_col: 'label', w1: 'weight1',  w2: 'weight2'
                }
            )
            
            if score_scale :
                data['score'] = data['score'] * score_scale
                
            # Compute scores by every group     
            values = list(set(data[break_col].values.flatten().tolist()))
            
            for val in break_val :
                # break down
                df_scr = data.loc[data[break_col]==val, :]
                
                # filter 
                for fv in fval :
                    df_scr1 = df_scr.loc[fops(df_scr[filter_col],fv), :].copy()

                    if w1 is None :
                        df_scr1['weight1'] = 1
                    if w2 is None :
                        df_scr1['weight2'] = 1

                    scores.append(
                        df_scr1[df_scr1['score'].notnull()]
                    )
            self.scores = scores
            return scores

    def score_cut(self, cut_step=0.02, buckets=100):
        '''
        Must run after self.run_score. cutoff and operating point will be generated. 
        Pre-assume the score is in interval [0,1], both result will be produced and saved in self attributes.
        cut_step: the step length by equal-width cut
        buckets: number of bins by equal-freq cut
        '''
        self.cutset = []
        self.opset  = []
        
        for eval_df in self.scores :
            # Evaluation on score cutoff
            cutoff = np.arange(0, 1, cut_step)
            pre_cut = []
            for c in cutoff :
                topc = eval_df.loc[eval_df['score'] > c, :]
                pre = topc.loc[topc['label']==1, 'weight1'].sum() / topc['weight1'].sum()
                rec = topc.loc[topc['label']==1, 'weight2'].sum() / (eval_df.loc[eval_df['label'] == 1, 'weight2']).sum()
                fpr = topc.loc[topc['label']==0, 'weight2'].sum() / (eval_df.loc[eval_df['label'] == 0, 'weight2']).sum()
                cut = c
                
                rlt0 = [pre, rec, fpr, cut, topc.shape[0], topc.loc[topc['label'] == 1, 'weight1'].sum(), topc.loc[topc['label']==1, 'weight2'].sum()]
                pre_cut.append(rlt0)
            pre_cut = pd.DataFrame.from_records(pre_cut, columns = ['precision', 'recall', 'fpr', 'cutoff', 'captured', '1_captured', 'w2_captured']).reset_index()
            self.cutset.append(pre_cut)
            
            # Evaluation on Operation point. 
            ops = range(0, eval_df.shape[0], eval_df.shape[0] // buckets)
            eval_df1 = eval_df.sort_values('score', ascending=False)
            pre_df = [] # pd.DataFrame()
            for i in ops :
                if i == 0 :
                    continue
                topi = eval_df1.iloc[:i,]
                pre  = topi.loc[topi['label'] == 1, 'weight1'].sum() / topi['weight1'].sum()
                rec  = topi.loc[topi['label'] == 1, 'weight2'].sum() / (eval_df1.loc[eval_df1['label'] == 1, 'weight2']).sum()
                fpr  = topi.loc[topi['label']==0, 'weight2'].sum() / (eval_df.loc[eval_df['label'] == 0, 'weight2']).sum()
                
                cut  = topi['score'].min()
                rlt0 = [pre, rec, fpr, cut, i, topi.loc[topi['label'] == 1, 'weight1'].sum(), topi.loc[topi['label']==1, 'weight2'].sum()]
                
                pre_df.append(rlt0)
            pre_df = pd.DataFrame.from_records(pre_df)
            pre_df.columns=['precision', 'recall', 'fpr', 'cutoff', 'captured', '1_captured', 'w2_captured']
            pre_df = pre_df.reset_index()
            
            self.opset.append(pre_df)
            
    def daily_recall(self, figsize=(16,8), qtl_list = [99, 95, 90, 50,], x_step=2):
        
        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))
        
        for ax, data in zip(axes, self.scores) :
        
            data1 = data.copy()

            for q in qtl_list :
                # tp : true positive
                data1[f'tp{q}'] = data1.apply(lambda x : x['label']==1 and x['score']>=q/100, axis=1)
                # cp : conditional positive
                data1[f'cp{q}'] = data1.apply(lambda x : x['label']==1, axis=1)

            stats_bydate = data1.groupby('date').agg(np.sum).reset_index()
            
            for q in qtl_list :
                stats_bydate[f'r{q}'] = stats_bydate.apply(lambda x : x[f'tp{q}'] / (x[f'cp{q}'] + 0.0001), axis=1)

            ax.bar(stats_bydate['date'], stats_bydate['label'], color = 'lightsteelblue')

            date_axis = stats_bydate['date']
            
            ax.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])
            plt.xticks(rotation=90)
            
            ax2 = ax.twinx()

            ax2.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax2.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])
            plt.xticks(rotation=90)

            date_axis = stats_bydate['date']

            handles = []
            for q in qtl_list :

                hdl, = ax2.plot(stats_bydate['date'], stats_bydate[q])
                handles.append(hdl)
            
            ax2.set_yticks(np.arange(0,1.02,0.02))
            ax2.grid(axis='y', linestyle='--')

            plt.legend(handles=handles, labels=qtl_list, loc='best')            
        plt.show()
        
    def daily_precision(self, figsize=(16,8), qtl_list = [99, 95, 90, 50,], x_step=2):
        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))

        for ax, data in zip(axes, self.scores) :
            data1 = data.copy()
           
            for q in qtl_list :
                # tp : true positive
                data1[f'tp{q}'] = data1.apply(lambda x : x['label']==1 and x['score']>=q/100, axis=1)
                # pp : predicted positive
                data1[f'pp{q}'] = data1.apply(lambda x : x['score']>=1/100, axis=1)

            stats_bydate = data1.groupby('date').agg(np.sum).reset_index()
            
            for q in qtl_list :
                stats_bydate[f'r{q}'] = stats_bydate.apply(lambda x : x[f'tp{q}'] / (x[f'pp{q}'] + 0.0001), axis=1)
            
            stats_bydate = data1.groupby('date').agg(pd.Series.sum).reset_index()
            
            for q in qtl_list :
                stats_bydate[f'p{q}'] = stats_bydate.apply(lambda x : x[f'tp{q}'] / (x[f'pp{q}'] + 0.0001), axis=1)
            
            ax.bar(stats_bydate['date'], stats_bydate['label'], color = 'lightsteelblue')

            date_axis = stats_bydate['date']
            ax.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])
         
            plt.xticks(rotation=90)
            
            ax2 = ax.twinx()

            ax2.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax2.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])

            plt.xticks(rotation=90)

            handles = []
            for q in qtl_list :

                hdl, = ax2.plot(stats_bydate['date'], stats_bydate[q])
                handles.append(hdl)
            
            ax2.set_yticks(np.arange(0,1.02,0.02))
            ax2.grid(axis='y', linestyle='--')

            plt.legend(handles=handles, labels=qtl_list, loc='best')            
        plt.show()


    def daily_qtls(self, figsize=(16,8), qtl_list = [99,90,50,10,1], x_step=2 ) :
        
        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))
        for ax, data in zip(axes, self.scores) :
            stats_bydate = data.groupby('date').agg({
                'label': pd.Series.sum, 
                'score': lambda x: np.quantile(x, q=[x / 100 for x in qtl_list], )
            }).reset_index()
            stats_bydate.columns = [
                'date',
                'label_cnt', 
            ] + [ f'q{x}' for x in qtl_list ]
            
            ax.bar(stats_bydate['date'], stats_bydate['label_cnt'], color = 'lightsteelblue')

            date_axis = stats_bydate['date'].unique()
            
            ax.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])
         
            plt.xticks(rotation=90)
            
            ax2 = ax.twinx()

            ax2.set_xticks([ i for i in range(0, len(date_axis), x_step) ])
            ax2.set_xticklabels([ date_axis[i] for i in range(0, len(date_axis), x_step) ])

            plt.xticks(rotation=90)

            handles = []
            for q in qtl_list :

                hdl, =ax2.plot(stats_bydate['date'], stats_bydate[q])
                handles.append(hdl)

            ax2.set_yticks(np.arange(0,1.02,0.02))
            ax2.set_ylim(0,1.02)
            ax2.grid(axis='y', linestyle='--')


            plt.legend(handles=handles, labels=qtl_list, loc='best')            
        plt.show()
    
    def plot_iv(self, bins=20, figsize=(10, 10), bin_col='score_bin', save_path=None) :

        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))
        
        for ax, data, model in zip(axes, self.scores, self.models) :
            # score distribution
            data_final_1 = data.assign(
                score_bin = pd.cut(data['score'], bins),
                score_qtl = pd.qcut(data['score'], bins),
            )

            N0 = (data_final_1['label'] == 0).sum()
            N1 = (data_final_1['label'] == 1).sum()

            
            data_final_2 = data_final_1.groupby(bin_col).agg({
                'label' : [pd.Series.count, pd.Series.sum]
            })

            data_final_2.columns = ['cnt', 'inc']

            data_final_2 = data_final_2.reset_index()
            data_final_2 = data_final_2.assign(
                woe = lambda x: np.log(x['inc'] / N1) - np.log((x['cnt'] - x['inc'] ) / N0),
                binbd = data_final_2[bin_col].apply(lambda x: x.right),
            )

            score_iv = np.sum((data_final_2['inc'] / N1 - (data_final_2['cnt'] - data_final_2['inc'])/ N0) * data_final_2['woe'])
            
            self.ivset.append(score_iv)
        
            cmap = self.cmap
            ax.bar( data_final_2.index, data_final_2['woe'], color=(data_final_2['woe']<=0).astype(int).apply(lambda x: cmap[x]))
            ax.set_xticklabels(data_final_2['binbd'])
            ax.set_title(f'{model.name} IV:{score_iv}')

            self._save(fig, plt, save_path)

    def plot_histogram(self, bins=20, bin_col='score_bin', label_col='label', plot=True, figsize=(10, 10), save_path=None) :

        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))
        
        for ax, score, model in zip(axes, self.scores, self.models) :
            ax.hist(score[score[label_col]==0], bins, density=True, histtype='bar', align='mid', rwidth=100//bins, alpha=0.75)
            ax.hist(score[score[label_col]==1], bins, density=True, histtype='bar', align='mid', rwidth=100//bins, alpha=0.75)
            ax.set_title(f'{model.name}')

        self._save(fig, plt, save_path)

    def plot_distribution(self, distr='normal', bins=100, label_col='label', alpha=0.75, figsize=(10, 10), save_path=None) :

        fig, axes = self._initialize_plot(figsize=(figsize[0], figsize[1]*len(self.models)))
        
        for ax, score, model in zip(axes, self.scores, self.models) :
            df0, df1  = score.loc[score[label_col]==0,'score'], score.loc[score[label_col]==1,'score']
            mu0, sig0 = df0.mean(), df0.std() 
            mu1, sig1 = df1.mean(), df1.std()
            if distr == 'normal' :
                spl0 = np.random.normal(mu0, sig0, 20000)
                spl1 = np.random.normal(mu1, sig1, 20000)
            else :
                raise ValueError("Only normal distribution is supported in current version.")
            
            _,_, p0 = ax.hist(spl0, bins, density=True, histtype='bar', align='mid', rwidth=bins, alpha=alpha, color=self.cmap[0])
            _,_, p1 = ax.hist(spl1, bins, density=True, histtype='bar', align='mid', rwidth=bins, alpha=alpha, color=self.cmap[1])

            ax.set_title(f'{model.name}')
            
            ax.legend(handles=[p0,p1],labels=['label-0', 'label-1'], loc='best')
        self._save(fig, plt, save_path)

    def plot_cutoff_chart(self, index_col=None, x_step=2, xlim=(0,100), ylim=(0,1.02), save_path=None, figsize=(10,10), chunksize=None) :   
        
        chunksize = len(self.opset) if chunksize is None else chunksize
        
        for i in range(0, len(self.opset), chunksize) :
            
            md_names = [ m.name for m in self.models[i:i+chunksize] ]
            fig, ax = plt.subplots(1,1, figsize=figsize)
            ax.ylim=ylim
            ax.y_ticks=ylim

            ax2 = ax.twinx()
            handles = []
            for pre_df, c in zip(self.cutset[i:i+chunksize], self.cmap[:chunksize]) :

                pre_df = pre_df.head(xlim[1])

                p1, = ax.plot(pre_df.loc[:, 'precision'].fillna(0), color=c)

                p2, = ax2.plot(pre_df.loc[:, 'recall'].fillna(0),  linestyle='--', color=c)

                handles.extend([p1, p2])

            ax.set_ylabel('precision')
            ax.set_xlim(*xlim)
            ax.set_xticks(range(0, pre_df.shape[0], x_step))
            ax.grid(which='both', axis='both')

            ax2.set_ylabel('recall')
            ax2.set_xlim(*xlim)
            ax2.grid(which='both', axis='both', linestyle='--')


            plt.legend(handles=handles, labels=product(md_names, ['precision', 'recall']), loc='best')            

            if index_col:
                ax.set_xticklabels(pre_df[index_col].map(lambda x: round(x,2)).iloc[0: pre_df.shape[0]+1:x_step])
            plt.xticks(rotation=90)

            self._save(fig, plt, save_path)
        
    def plot_op_chart(self, index_col=None, x_step=2, xlim=(0,100), ylim=(0,1.02), save_path=None, figsize=(10,10), chunksize=None) :
        
        chunksize = len(self.opset) if chunksize is None else chunksize
        
        for i in range(0, len(self.opset), chunksize) :
            
            md_names = [ m.name for m in self.models[i:i+chunksize] ]
            fig, ax = plt.subplots(1,1, figsize=figsize)
            ax.ylim=ylim
            ax.y_ticks=ylim

            ax2 = ax.twinx()
            handles = []
            for pre_df, c in zip(self.opset[i:i+chunksize], self.cmap[:chunksize]) :

                pre_df = pre_df.head(xlim[1])

                p1, = ax.plot(pre_df.loc[:, 'precision'].fillna(0), color=c)

                p2, = ax2.plot(pre_df.loc[:, 'recall'].fillna(0),  linestyle='--', color=c)

                handles.extend([p1, p2])

            ax.set_ylabel('precision')
            ax.set_xlim(*xlim)
            ax.set_xticks(range(0, pre_df.shape[0], x_step))
            ax.grid(which='both', axis='both')

            ax2.set_ylabel('recall')
            ax2.set_xlim(*xlim)
            ax2.grid(which='both', axis='both', linestyle='--')


            plt.legend(handles=handles, labels=product(md_names, ['precision', 'recall']), loc='best')            

            if index_col:
                ax.set_xticklabels(pre_df[index_col].map(lambda x: round(x,2)).iloc[0: pre_df.shape[0]+1:x_step])
            plt.xticks(rotation=90)

            self._save(fig, plt, save_path)

    def plot_pr(self, tick_step=0.02, save_path=None, figsize=(10,10)) :
        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
        ax.ylim=(0,1)
        ax.y_ticks=(0,1)
        
        handles = []
        for i, pre_df in enumerate(self.opset) :
            p1, = ax.plot(
                pre_df.loc[(~pre_df['recall'].isnull()) & (pre_df['recall']>0.0), 'recall'],
                pre_df.loc[(~pre_df['precision'].isnull()) & (pre_df['precision']>0.0), 'precision'],
                color=self.cmap[i],
            )
            handles.append(p1)
            
        ax.set_ylabel('precision')
        ax.set_xlabel('recall')
        ax.set_xlim(0,1.02)
        ax.set_ylim(0,1.02)
        ax.grid(which='both', axis='both')
        
        plt.legend(handles=handles, labels=[ m.name for m in self.models ], loc='best')     
        
        self._save(fig, plt, save_path)

    def plot_roc(self, tick_step=0.02, save_path=None, figsize=(10,10)) :
        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
        handles = []
        for i, pre_df in enumerate(self.opset) :
            p1, = ax.plot(
                pre_df.loc[(~pre_df['fpr'].isnull()) & (pre_df['fpr']>0.0), 'fpr'],
                pre_df.loc[(~pre_df['recall'].isnull()) & (pre_df['recall']>0.0), 'recall'],
                color=self.cmap[i],
            )
            handles.append(p1)
            
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        ax.set_xlim(0,1.02)
        ax.set_ylim(0,1.02)
        ax.grid(which='both', axis='both')
        
        plt.legend(handles=handles, labels=[ m.name for m in self.models ], loc='best')     
        
        self._save(fig, plt, save_path)

class ScoreEvalNegative(ScoreEval) :

    def score_cut(self, cut_step=0.02, buckets=100):
        '''
        Must run after self.run_score. cutoff and operating point will be generated. 
        Pre-assume the score is in interval [0,1], both result will be produced and saved in self attributes.
        cut_step: the step length by equal-width cut
        buckets: number of bins by equal-freq cut
        '''
        self.cutset = []
        self.opset  = []
        
        for eval_df in self.scores :
            # Evaluation on score cutoff
            cutoff = np.arange(0, 1, cut_step)
            pre_cut = []
            for c in cutoff :
                topc = eval_df.loc[eval_df['score'] < c, :]
                pre = topc.loc[topc['label']==1, 'weight1'].sum() / topc['weight1'].sum()
                rec = topc.loc[topc['label']==1, 'weight2'].sum() / (eval_df.loc[eval_df['label'] == 1, 'weight2']).sum()
         
                cut = c
                
                rlt0 = [pre, rec, cut, topc.shape[0], (topc['label'] == 1).sum(), (topc.loc[topc['label']==1, 'weight2'].sum())]
                pre_cut.append(rlt0)
            pre_cut = pd.DataFrame.from_records(pre_cut, columns = ['precision', 'recall', 'cutoff', 'captured', '1_captured', 'w2_captured']).reset_index()
            self.cutset.append(pre_cut)
            
            # Evaluation on Operation point. 
            ops = range(0, eval_df.shape[0], eval_df.shape[0] // buckets)
            eval_df1 = eval_df.sort_values('score', ascending=True)
            pre_df = [] # pd.DataFrame()
            for i in ops :
                if i == 0 :
                    continue
                topi = eval_df1.iloc[:i,]
                pre  = (topi.loc[topi['label'] == 1, 'weight1']).sum() / topi['weight1'].sum()
                
                rec  = (topi.loc[topi['label'] == 1, 'weight2']).sum() / (eval_df1.loc[eval_df1['label'] == 1, 'weight2']).sum()
                
                
                cut  = topi['score'].max()
                rlt0 = [pre, rec, cut, i, (topi['label'] == 1).sum(), (topi.loc[topi['label']==1, 'weight2'].sum())]
                
                pre_df.append(rlt0)
            pre_df = pd.DataFrame.from_records(pre_df)
            pre_df.columns=['precision', 'recall', 'cutoff', 'captured', '1_captured', 'w2_captured']
            pre_df = pre_df.reset_index()
            
            self.opset.append(pre_df)


if __name__ == '__main__' :
    
    from sklearn.linear_model import LogisticRegression
    BASE_DIR = "E:\\workspace\\scoreval"
    data = pd.read_csv(os.path.join(BASE_DIR,"data/investing_program_prediction_data.csv"))
    # Build simple model for test
    features = ['SE1', 'BA1', 'BA2', 'BA3', 'BA4', 'BA5']
    label = 'label'
    data['label'] = data['InvType'].map(lambda x: x[-1]).astype(int)
    # normalize
    for f in features : 
        data[f] = (data[f] - data[f].mean()) / data[f].std()
    X = data[features].values
    y = data[label].values

    models = []

    for i in range(1,4) :
        clf = LogisticRegression(
            penalty='l2', #'elasticnet',
            # l1_ratio=0.5,
            tol=1e-6,
            solver='saga',
            max_iter=200,
            C=i/100,

        ).fit(X, y)
        # clf.predict(X[:2, :])
        # clf.score(X,y)
        clf.name = f'model{i}'
        models.append(clf)

    # Create ScoreEval object, initialized by a list of model objects
    se = ScoreEval(models)
    def skl_predict(clf, X) :
        pred = clf.predict_proba(X)
        return pred[:,1].reshape(pred.shape[0], 1)
    # Run score, make sure the models have the same inputs. SE will call model.predict for each model. 
    # That is not a good idea to take models with different inputs. Please always compare apple to apple. 
    # Make sure X and Y have the same rows so that the score can be produced properly. 
    # se_oot.run_score([oot_X_arr,oot_S_arr,oot_P_arr,oot_I_arr,], df_y)
    se.run_score(X, data[[label]], skl_predict)
    se.score_cut(0.01,100)
    se.plot_cutoff_chart(xlim=(0, 50), save_path='./cutoff.png')
    se.plot_op_chart(xlim=(0, 50), save_path='./op.png')
    se.plot_pr(save_path='./pr.png')
    se.plot_roc(save_path='./roc.png')
    se.plot_iv(save_path='./iv.png')
    se.plot_histogram(save_path='./hist.png')
    se.plot_distribution(save_path='./distr.png')