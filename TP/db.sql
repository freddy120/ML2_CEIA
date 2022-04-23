create database ml2;

CREATE TABLE "heart_table" (
  "id" BIGSERIAL PRIMARY KEY,
  "Age" int,
  "Sex" varchar(255),
  "ChestPainType" varchar(10),
  "RestingBP" int,
  "Cholesterol" int,
  "FastingBS" int,
  "RestingECG" varchar(255),
  "MaxHR" int,
  "ExerciseAngina" varchar(255),
  "Oldpeak" float,
  "ST_Slope" varchar(255),
  "HeartDisease" int
);


CREATE TABLE "heart_predictions" (
  "id" BIGSERIAL PRIMARY KEY,
  "Age" int,
  "Sex" varchar(255),
  "ChestPainType" varchar(10),
  "RestingBP" int,
  "Cholesterol" int,
  "FastingBS" int,
  "RestingECG" varchar(255),
  "MaxHR" int,
  "ExerciseAngina" varchar(255),
  "Oldpeak" float,
  "ST_Slope" varchar(255),
  "HeartDisease" int
);



select * from heart_table;
select * from heart_predictions;