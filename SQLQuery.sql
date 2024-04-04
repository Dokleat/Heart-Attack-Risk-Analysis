--Number of patients by gender

SELECT Sex, COUNT(*) as Number_of_Patients
FROM heart_attack_prediction
GROUP BY Sex;

--Average cholesterol for patients with and without heart attack risk

SELECT Heart_Attack_Risk, AVG(Cholesterol) as Average_Cholesterol
FROM heart_attack_prediction
GROUP BY Heart_Attack_Risk;

--Number of patients by diabetes status and heart attack risk

SELECT Diabetes, Heart_Attack_Risk, COUNT(*) as Number_of_Patients
FROM heart_attack_prediction
GROUP BY Diabetes, Heart_Attack_Risk;

--Average sedentary hours per day by heart attack risk

SELECT Heart_Attack_Risk, AVG(Sedentary_Hours_Per_Day) as Average_Sedentary_Hours
FROM heart_attack_prediction
GROUP BY Heart_Attack_Risk;

--Distribution of patients by continent and risk of heart attack

SELECT Continent, Heart_Attack_Risk, COUNT(*) as Number_of_Patients
FROM heart_attack_prediction
GROUP BY Continent, Heart_Attack_Risk;

--Average age for patients by risk for heart attack and diabetes

SELECT Heart_Attack_Risk, Diabetes, AVG(Age) as Average_Age
FROM heart_attack_prediction
GROUP BY Heart_Attack_Risk, Diabetes;

--Number of patients who drink and smoke according to heart attack risk

SELECT Smoking, Heart_Attack_Risk, COUNT(*) as Number_of_Patients
FROM heart_attack_prediction
WHERE Smoking = 1
GROUP BY Heart_Attack_Risk;

--Distribution of obese patients according to weekly physical activity

SELECT Physical_Activity_Days_Per_Week, Obesity, COUNT(*) as Number_of_Patients
FROM heart_attack_prediction
WHERE Obesity = 1
GROUP BY Physical_Activity_Days_Per_Week;

--Mean triglycerides for patients at high and low risk for heart attack

SELECT Heart_Attack_Risk, AVG(Triglycerides) as Average_Triglycerides
FROM heart_attack_prediction
GROUP BY Heart_Attack_Risk;

--Analysis of mean blood pressure in patients with a family history of heart disease

SELECT Family_History, 
       AVG(CAST(SUBSTRING(Blood_Pressure, 1, CHARINDEX('/', Blood_Pressure) - 1) AS INT)) as Average_Systolic_BP,
       AVG(CAST(SUBSTRING(Blood_Pressure, CHARINDEX('/', Blood_Pressure) + 1, LEN(Blood_Pressure)) AS INT)) as Average_Diastolic_BP
FROM heart_attack_prediction
WHERE Family_History = 1
GROUP BY Family_History;



