# coding=utf-8

# Authors: Mariana A. Souza <mas2@cin.ufpe.br>
#          Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import copy

from scipy.stats import mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from deslib.util.sgh import SGH, SGH2
from deslib.base import BaseDS
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.a_priori import APriori
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.mla import MLA
from deslib.util.instance_hardness import kdn_score




class OLP(BaseDS):
    """
    Online Local Pool (OLP).

    This technique dynamically generates and selects a pool of classifiers based on the local region each given
    query sample is located, if such region has any degree of class overlap. Otherwise, the technique uses the
    KNN rule for obtaining the query sample's label.

    Parameters
    ----------

    n_classifiers : int (default = 7)
             The size of the pool to be generated for each query instance.

    k : int (Default = 7)
        Number of neighbors used to obtain the Region of Competence (RoC).

    IH_rate : float (default = 0.0)
        Hardness threshold used to identify when to generate the local pool or not.

    ds_tech : str (default = 'ola')
        DCS technique to be coupled to the OLP.

    References
    ----------

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin, Online local pool generation for
    dynamic classifier selection, Pattern Recognition 85 (2019) 132-148.

    """

    def __init__(self, n_classifiers=7, k=7, IH_rate=0.0, ds_tech='ola', knne=False, sgh = SGH):

        super(OLP, self).__init__([], k, IH_rate=IH_rate)

        self.name = 'Online Local Pool (OLP)'
        self.ds_tech = ds_tech
        self.n_classifiers = n_classifiers
        self.knne_ = knne
        self.sgh = sgh

    def fit(self, X, y):
        """
        Prepare the model by setting the KNN algorithm and
        calculates the information required to apply the OLP

        Parameters
        ----------

        X : array of shape = [n_samples, n_features]
            The input data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Label encoder
        self.base_already_encoded_ = False
        self._setup_label_encoder(y)
        y_ind = self._encode_base_labels(y)

        self._set_dsel(X, y_ind)

        if self.k > self.n_samples_:
            self.k = self.n_samples_ - 1

        # Set self.knn_class_ 
        self._set_region_of_competence_algorithm() #set KNN or FAISS or an estimator passed as parameter
        self._fit_region_competence(X, y_ind) # self.k is already set in set_region_of_competence self._fit_region_competence(X, y_ind, self.k) #train the KNN or FAISS or estimator using training data

        # Calculate the KDN score of the training samples
        self.hardness_, _ = kdn_score(X, y_ind, self.k) #array same size as y

        return self

    def _set_dsel(self, X, y):
        """
        Get information about the structure of the data (e.g., n_classes, N_samples, classes)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        Returns
        -------
        self
        """
        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.n_samples_ = self.DSEL_target_.size

        return self

    def _generate_local_pool(self, query):
        """
        Local pool generation.

        This procedure populates the "pool_classifiers" based on the query sample's neighborhood.
        Thus, for each query sample, a different pool is created.

        In each iteration, the training samples near the query sample are singled out and a
        subpool is generated using the Self-Generating Hyperplanes (SGH) method.
        Then, the DCS technique selects the best classifier in the generated subpool.
        In the following iteration, the neighborhood is increased and another SGH-generated subpool is obtained
        over the new neighborhood, and again the DCS technique singles out the best in it,
        which is then added to the local pool. This process is repeated until the pool reaches "n_classifiers".

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample.

        Returns
        -------
        self

        References
        ----------
        Souza, Mariana A., George DC Cavalcanti, Rafael MO Cruz, and Robert Sabourin. "Online local pool generation
        for dynamic classifier selection." Pattern Recognition 85 (2019): 132-148.

        M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin, On the characterization of the
        oracle for dynamic classifier selection, in: International Joint Conference on Neural Networks,
        IEEE, 2017, pp. 332-339.
        """
        n_samples, _ = self.DSEL_data_.shape

        self.pool_classifiers = []

        n_err = 0
        max_err = 2 * self.n_classifiers

        curr_k = self.k

        # Classifier count
        n = 0
        # Adding condition (curr_k <= self.n_samples) to handle special cases where curr_k can be higher than n_samples.
        while (n < self.n_classifiers and
               n_err < max_err and
               curr_k <= self.n_samples_):

            subpool = self.sgh()

            included_samples = np.zeros(n_samples, dtype=int)

            if self.knne_:
                idx_neighb = np.array([], dtype=int)

                #Obtain neighbors of each class individually
                # for j in np.arange(0, self.n_classes_):
                #     # Obtain neighbors from the classes in the RoC
                #     if np.any(self.classes_[j] == self.DSEL_target_[self.neighbors[0][np.arange(0, curr_k)]]):
                #         nc = np.where(self.classes_[j] == self.DSEL_target_[self.neighbors[0]])
                #         idx_nc = self.neighbors[0][nc]
                #         idx_nc = idx_nc[np.arange(0, np.minimum(curr_k, len(idx_nc)))]
                #         idx_neighb = np.concatenate((idx_neighb, idx_nc), axis=0)
                
                for enc_class in self.enc_.transform(self.classes_):
                    # Obtain neighbors from the classes in the RoC
                    if np.any(enc_class == self.DSEL_target_[self.neighbors[0][np.arange(0, curr_k)]]): #the 0 in self.neighbors[0] is just to inside like when using np.where. From the closest curr_k sample, is there any with enc_class?
                        nc = np.where(enc_class == self.DSEL_target_[self.neighbors[0]]) #obs: DSEL_target indexes is different from self.DSEL_target_[self.neighbors[0]]
                        idx_nc = self.neighbors[0][nc] #Gets the closest neighbors to the sample that have enc_class label
                        idx_nc = idx_nc[np.arange(0, np.minimum(curr_k, len(idx_nc)))] # get only curr_k or less
                        idx_neighb = np.concatenate((idx_neighb, idx_nc), axis=0)

            else:
                idx_neighb = np.asarray(self.neighbors)[0][np.arange(0, curr_k)]

            # Indicate participating instances in the training of the subpool
            included_samples[idx_neighb] = 1

            curr_classes = np.unique(self.DSEL_target_[idx_neighb])

            # If there are +1 classes in the local region
            if len(curr_classes) > 1:
                # Obtain SGH pool
                subpool.fit(self.DSEL_data_, self.DSEL_target_, included_samples)

                # Adjust chosen DCS technique parameters
                if self.ds_tech == 'ola': #TODO: investigate if should use self.k because in case of KNNE when k =3 and dataset has 2 classes, len(idx_neighb) is equal to 6
                    ds = OLA(subpool, k=len(idx_neighb))  # change for self.k
                elif self.ds_tech == 'lca':
                    ds = LCA(subpool, k=len(idx_neighb))
                elif self.ds_tech == 'mcb':
                    ds = MCB(subpool, k=len(idx_neighb))
                elif self.ds_tech == 'mla':
                    ds = MLA(subpool, k=len(idx_neighb))
                elif self.ds_tech == 'a_priori':
                    ds = APriori(subpool, k=len(idx_neighb))
                elif self.ds_tech == 'a_posteriori':
                    ds = APosteriori(subpool, k=len(idx_neighb))

                # Fit ds technique
                ds.fit(self.DSEL_data_, self.DSEL_target_)

                neighb = np.in1d(self.neighbors, idx_neighb)  # True/False vector of selected neighbors

                # Set distances and neighbors of the query sample (already calculated)
                ds.distances = np.asarray([self.distances[0][neighb]])  # Neighborhood
                ds.neighbors = np.asarray([self.neighbors[0][neighb]])  # Neighborhood

                ds.DFP_mask = np.ones(ds.n_classifiers_)

                # Estimate competence
                comp = ds.estimate_competence(query, ds._predict_base(query))

                # Select best classifier in subpool
                sel_c = ds.select(comp)

                # Add to local pool
                self.pool_classifiers.append(copy.deepcopy(subpool[sel_c[0]]))

                n += 1

            # Increase neighborhood size
            curr_k += 2
            n_err += 1

        # Handle cases where not all classifiers are generated due to the lack of samples in DSEL
        self.n_classifiers_ = n
        return

    def select(self, query):
        """
        Obtains the votes of each classifier given a query sample.

        Parameters
        ----------
        query : array of shape = [n_features] containing the test sample

        Returns
        -------
        votes : array of shape = [len(pool_classifiers)] with the class yielded by each classifier in the pool

        """

        votes = np.zeros(len(self.pool_classifiers), dtype=int)
        for clf_idx, clf in enumerate(self.pool_classifiers):
            votes[clf_idx] = clf.predict(query)[0]

        return votes

    def classify_with_ds(self, query, predictions=None, probabilities=None):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)

        # Predict query label
        if len(self.pool_classifiers) > 0:
            votes = self.select(query)
            predicted_label = mode(votes, keepdims=False)[0]
        else:
            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]
            predicted_label = mode(self.DSEL_target_[roc], keepdims=False)[0]

        return predicted_label

    def predict_proba_with_ds(self, query, predictions=None, probabilities=None):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)
        probs = np.zeros(self.n_classes_)
        # Predict query label
        if len(self.pool_classifiers) > 0:
            votes = self.select(query)
            for idx in range(self.n_classes_):
                probs[idx] = np.sum(votes == idx) / self.n_classifiers_
        else:
            nn = np.arange(0, self.k)
            votes = self.neighbors[0][nn]
            for idx in range(self.n_classes_):
                probs[idx] = np.sum(votes == idx) / self.k

        return probs
    
    def _predict(self, 
                  X, 
                  knn_predictions, 
                  final_predictions, 
                  fn_predict, 
                  fn_ds_predict):
        """
        Predicts the class label for each sample in X.
    
        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.
    
        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if the DS model was trained
        check_is_fitted(self, ["DSEL_data_", "DSEL_target_", "hardness_"])
        
    
        # Get the region of competence for all instances in X
        tmp_k = np.minimum(self.n_samples_, self.n_classes_ * self.n_classifiers * self.k)
        distances, neighbors = self._get_region_competence(X, k=tmp_k) #TODO: pass as parameters
    
        # Extract the nearest neighbors for each instance
        nn = np.arange(0, self.k)
        roc = neighbors[:, nn]
    
        # Find instances with all neighbors having Instance Hardness (IH) below or equal to IH_rate
        ih_threshold_met = np.all(self.hardness_[roc] <= self.IH_rate, axis=1)
        
        # Use KNN for instances meeting the IH threshold
        knn_indices = np.where(ih_threshold_met)[0]
        if knn_indices.size:
            knn_predictions[knn_indices] = fn_predict(X[knn_indices])
    
        # Use DS for instances not meeting the IH threshold
        ds_indices = np.where(~ih_threshold_met)[0]
        ds_predictions = []
        for idx in ds_indices:
            self.distances = np.atleast_2d(distances[idx])
            self.neighbors = np.atleast_2d(neighbors[idx])
            l = fn_ds_predict(np.atleast_2d(X[idx]))
            ds_predictions.append(l)
    
        final_predictions[knn_indices] = knn_predictions[knn_indices]
        final_predictions[ds_indices] =  ds_predictions
        
        self.neighbors = None
        self.distances = None
        
        return final_predictions 
    

    def predict(self, X):
       """
       Predicts the class label for each sample in X.
   
       Parameters
       ----------
       X : array of shape = [n_samples, n_features]
           The input data.
   
       Returns
       -------
       predicted_labels : array of shape = [n_samples]
                          Predicted class label for each sample in X.
       """
       # Check if X is a valid input
       X = check_array(X)
       knn_predictions = np.zeros(X.shape[0], dtype=int)
       final_predictions = np.empty(X.shape[0], dtype=int)
       fn_predict = self.roc_algorithm_.predict
       fn_ds_predict = self.classify_with_ds
       
       predictions = self._predict(X, 
                             knn_predictions, 
                             final_predictions, 
                             fn_predict, 
                             fn_ds_predict)
       
       
       return self.classes_.take(predictions)
   
    def predict_proba(self, X):
        """
        Predicts the class label for each sample in X.
    
        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.
    
        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if X is a valid input
        X = check_array(X)
        knn_predictions = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        final_predictions = np.empty((X.shape[0], self.n_classes_), dtype=float)
        fn_predict = self.roc_algorithm_.predict_proba
        fn_ds_predict = self.predict_proba_with_ds
        
        return self._predict(X, 
                              knn_predictions, 
                              final_predictions, 
                              fn_predict, 
                              fn_ds_predict)