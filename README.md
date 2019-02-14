# Machine-learning-in-Mammographic-Mass-Data

A repository to store progress on ML project.

This playground can be mimic using ML to predict the severity (benign or malignant) of a mammographic mass lesion from BI-RADS attributes (our project has 3 features ['Shape','Margin',"Density"])and the patient's age.The ML model will be deployed to Heroku as a Flask app [viewable here](). The top 2 winner of all our tested ML models (LogisticRegression and SVC kernal rbf)will be used  to predit.
Available models comparation: 
<p align="center">
  <img src="https://github.com/vivianccpan/Machine-learning-in-Mammographic-Mass-Data/blob/master/all_ML_models_validation.png" alt="Process Diagram"/>
</p>


All our model is base on the dataset from the UCI repository [source viewable here]: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass).Also thank for https://sundog-education.com/ course support.

BI-RADS attributes resource can be access [here] http://www.radiologyassistant.nl/en/p53b4082c92130/bi-rads-for-mammography-and-ultrasound-2013.html


> Additional data visualization is hosted on Tableau Public [viewable here](https://public.tableau.com/views/MammographicMassDataML_preview/Dashboard1?:embed=y&:display_count=yes&publish=yes).
<p align="center">
  <img src="https://github.com/vivianccpan/Machine-learning-in-Mammographic-Mass-Data/blob/master/Dashboard%201_ML.png" alt="Process Diagram"/>
</p>
