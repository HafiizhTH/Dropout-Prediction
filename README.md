
## Final Submission: Solving Educational Institution Problems

## Business Understanding

Jaya Jaya Institute is one of the educational institutions that has been established since 2000. To date it has produced many graduates with an excellent reputation.

### Business Problems

The problem faced by Jaya Jaya Institution is the high number of students who do not complete their education, aka dropouts. This high number of dropouts is one of the major problems for an educational institution. Therefore, Jaya Jaya Institute wants to detect potential dropout students as soon as possible in order to provide specialized guidance.

### Project Scope

The focus of this project is to solve a business problem with the following steps:
1. Identifying factors that affect student dropout.
2. Creating a business dashboard to monitor and provide insights to the institute regarding the student dropout problem.
3. Build a machine learning model to assist the institute in predicting student dropouts, so that it can take proactive action against students who are indicated not to complete school.

### Preparation

The dataset used in this project is sourced from dicoding. You can download the dataset by clicking the following link: [click here](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance).

To get started with our project, follow these installation steps:
- Clone the project repository from our gitHub repository:
```
git clone https://github.com/HafiizhTH/Dropout-Prediction.git
```

Setup environment - Anaconda:
- Open Anaconda Terminal
- Run the following command to create a new environment:
```
conda create --name main-ds python=3.11
```
- Activate the virtual environment by running the following command:
```
conda activate main-ds
```
- Install the library to use:
```
pip install -r requirements.txt
```
Run streamlit app
```
streamlit run app.py
```

## Business Dashboard

A business dashboard was created to monitor the factors that influence student dropout. The dashboard provides comprehensive data visualization, enabling institutions to provide appropriate and swift actions based on the insights gained. If you would like to access the dashboard [click here](https://lookerstudio.google.com/u/0/reporting/906a8401-a1eb-4f20-ac6a-fda975851a7c/page/CRG1D).

![image](https://github.com/HafiizhTH/Dropout-Prediction/assets/96015981/0ed46533-f9df-4812-9707-2ca134af2247)

In the Dashboard visualization, there are several features to analyze students who drop out of school, such as::
- **Filter and Data Control**: serves to facilitate the search and analysis of specific data based on student status and Attedance.
- **View Total**: serves to see the number and average number of students dropping out of school, the dropout rate, and the unemployment rate.
- **Pie Chart**: Used to see the comparison and percentage of data based on student status.
- **Bar Chart**: Used to view data distribution based on international status, scholarship recipients, tuition fee compliance, students with disabilities, displaced students, father's occupation, mother's occupation.
- **Table**: It serves to view detailed information about students by age, subject, grade, and curriculum. It can be used as an alternative to analyze dropout rates in depth.

## Testing Model

If you want to try using this model to predict student dropout. [click here](https://dropout-prediction-2024.streamlit.app/)

![image](https://github.com/HafiizhTH/Dropout-Prediction/assets/96015981/00fd7c96-7952-40b8-abed-f447546dd0ec)

In the testing view of the dropout student prediction model there are several pages that you can use, such as:
- **Prediction**: There are 2 ways that you can use, namely single-predict (manual input) or multiple-predict (dataset upload).
- **Student Information**: There are 2 ways you can use to view data that has been uploaded, namely viewing from descriptive data or visualization data.
- **FAQ**: Contains a list of questions and answers to help you run this application.

## Conclusion

Based on the analysis that has been carried out, several conclusions can be drawn as follows:
1. **Number of Dropout Students**: A total of 1,298 students dropped out, which is a very high number as it is almost 50% of the total 2,118 students who have graduated.
2. **Gender of Dropout Students**: Dropouts were more prevalent among male students compared to female students.
3. **Cost of Education**: Tuition fee inappropriateness is one of the main factors causing dropout, with 891 students finding that the tuition fees they face are inappropriate or too high.
4. **Neglected Students**: Displaced students are also a significant factor, with 619 students falling into this category.
5. **Parent Occupation**: Dropouts were most prevalent among students whose parents were not or had not worked. There were 424 students whose fathers did not or had not worked, and 584 students whose mothers did not or had not worked.
6. **Unemployment Rate**: The high unemployment rate of 11% also needs special attention. This suggests the need for further evaluation and action on the part of parents as well as educational institutions.

From these points, it is clear that various factors such as **education costs, family conditions and parental unemployment** have a significant influence on student dropout rates. Hence, a concerted effort on the part of the institute and parents is required to evaluate this issue.

### Recomandation

Based on the conclusions, the following recommendations can be implemented to overcome the dropout problem and improve the quality of education:
- **Scholarships and Education Subsidies**: The government and educational institutions can increase the number of scholarships and education subsidies to help students facing financial difficulties, especially for students whose families cannot afford to pay tuition fees, focusing on students whose parents are not or have not worked.
- **Mentoring and Guidance**: Organize guidance and counseling programs for students. especially for male students who are at risk of dropout because they are too high compared to females.
- **Training and Counseling**: Conduct training and counseling programs for parents to make them more aware of the importance of education and how to support their children.
- **Monitoring and Evaluation**: Create a monitoring and evaluation system that can detect students at risk of dropout, so that the institute can take action before they leave school.