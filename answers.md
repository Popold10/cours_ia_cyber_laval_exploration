\## Question 1: How many examples are there in the dataset?



Il y a 2494 exemple pour 28 colonnes.





\## Question 2: What is the distribution of the target?



Régions 	      Nombre de réponses

East North Central    758

West North Central    358

Middle Atlantic       334

South Atlantic        248

Pacific               243

Mountain              190

West South Central    172

East South Central     97

New England            94





\## Question 3: What are the features that can be used to predict the target?



\['RespondentID', 'What\_would\_you\_call\_the\_part\_of\_the\_country\_you\_live\_in\_now', 'How\_much\_do\_you\_personally\_identify\_as\_a\_Midwesterner', 'Do\_you\_consider\_Illinois\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Indiana\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Iowa\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Kansas\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Michigan\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Minnesota\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Missouri\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Nebraska\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_North\_Dakota\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Ohio\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_South\_Dakota\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Wisconsin\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Arkansas\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Colorado\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Kentucky\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Oklahoma\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Pennsylvania\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_West\_Virginia\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Montana\_state\_as\_part\_of\_the\_Midwest', 'Do\_you\_consider\_Wyoming\_state\_as\_part\_of\_the\_Midwest', 'Gender', 'Age', 'Household\_Income', 'Education', 'In\_what\_ZIP\_code\_is\_your\_home\_located']



Les données utilisées pour définir la cible sont la position géographique (nord, sud, est, ouest...) où il vit dans pays, s'il s'identifie à une certaine position géographique précise (nord, sud, est, ouest...), s'il identifie un état à une certaine partie géographique (nord, sud, est, ouest...), le sexe de la cible, son âge, son revenu, son niveau d'éducation et son code postal.





\## Question 4: Are there any missing values in the dataset?



Oui il existe des valeur manquante dans le dataset, elles sont montré par la présence de "?" dans les résultats de "X\["Household\_Income"].unique()" "X\["Education"].unique()"





\## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?



La réponse la plus commune est "not at all"

How\_much\_do\_you\_personally\_identify\_as\_a\_Midwesterner

Not at all    965

A lot         697

Some          528

Not much      304

Name: count, dtype: int64





\## Question 6: Among the three models, which one has the best recall?



Le modèle qui obtient le meilleur recall est le Gradient Boosting



Gradient Boosting : 0.990

Random Forest : 0.924

Régression Logistique : 0





\## Question 7: Which model has the best practical application?



| Modèle                     | Accuracy | Precision | Recall |

| -------------------------- | -------- | --------- | ------ |

| Logistic Regression        | 0.552    | 0.0       | 0.0    |

| Random Forest              | 0.944    | 0.949     | 0.924  |

| Gradient Boosting (HistGB) | 0.977    | 0.961     | 0.990  |





| Modèle                     | TP (estimé) | TN (estimé) | FP (estimé) | FN (estimé) | Score pratique approximatif |

| -------------------------- | ----------- | ----------- | ----------- | ----------- | --------------------------- |

| Logistic Regression        | 0           | 552         | 448         | 1000        | Très faible (-)             |

| Random Forest              | 924         | 950         | 50          | 76          | Élevé                       |

| Gradient Boosting (HistGB) | 990         | 970         | 30          | 10          | Très élevé                  |





Gradient Boosting est le meilleur choix car il maximise les TP et TN en minimisant FP





\## Question 8: Which model generalizes the best?



Modèle	        		Score train	Score test	Écart train-test	

Logistic Regression		0,55		0,55		0,0	

Random Forest			0,99		0,94		0,05	

Gradient Boosting (HistGB)	0,98		0,97		0,01	



