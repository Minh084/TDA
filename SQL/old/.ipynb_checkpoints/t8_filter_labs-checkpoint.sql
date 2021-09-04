select *
from triage.triage_cohort_2019_all_labs
where 
(
(lab_name="Glucose by Meter")
OR
(lab_name="Sodium, Ser/Plas")
OR
(lab_name="Potassium, Ser/Plas")
OR
(lab_name="Calcium, Ser/Plas")
OR
(lab_name="Creatinine, Ser/Plas")
OR
(lab_name="Chloride, Ser/Plas")
OR
(lab_name="CO2, Ser/Plas")
OR
(lab_name="Anion Gap")
OR
(lab_name="Glucose, Ser/Plas")
OR
(lab_name="BUN, Ser/Plas")
OR
(lab_name="Platelet count")
OR
(lab_name="Hematocrit")
OR
(lab_name="Hemoglobin")
OR
(lab_name="WBC")
OR
(lab_name="RBC")
OR
(lab_name="MCV")
OR
(lab_name="MCHC")
OR
(lab_name="MCH")
OR
(lab_name="RDW")
OR
(lab_name="Magnesium, Ser/Plas")
OR
(lab_name="Albumin, Ser/Plas")
OR
(lab_name="Protein, Total, Ser/Plas")
OR
(lab_name="AST (SGOT), Ser/Plas")
OR
(lab_name="ALT (SGPT), Ser/Plas")
OR
(lab_name="Alk P'TASE, Total, Ser/Plas")
OR
(lab_name="Total Bilirubin, Ser/Plas")
OR
(lab_name="Neutrophil, Absolute")
OR
(lab_name="Basophil %")
OR
(lab_name="Eosinophil %")
OR
(lab_name="Neutrophil %")
OR
(lab_name="Monocyte %")
OR
(lab_name="Lymphocyte %")
OR
(lab_name="Monocyte, Absolute")
OR
(lab_name="Lymphocyte, Absolute")
OR
(lab_name="Basophil, Absolute")
OR
(lab_name="Eosinophil, Absolute")
OR
(lab_name="Globulin")
OR
(lab_name="INR")
OR
(lab_name="Prothrombin Time")
OR
(lab_name="Phosphorus, Ser/Plas")
OR
(lab_name="Base Excess, ISTAT")
OR
(lab_name="Lactate, Whole Bld")
OR
(lab_name="TCO2 (a), ISTAT")
OR
(lab_name="pCO2 (a), ISTAT")
OR
(lab_name="PH (a), ISTAT")
OR
(lab_name="PO2 (a), ISTAT")
OR
(lab_name="HCO3 (a), ISTAT")
OR
(lab_name="O2 Saturation, ISTAT")
OR
(lab_name="tCO2")
OR
(lab_name="HCO3")
OR
(lab_name="WBC, urine")
OR
(lab_name="RBC, urine")
OR
(lab_name="TROPONIN I")
OR
(lab_name="Base Excess (vt)")
OR
(lab_name="pH, urine")
OR
(lab_name="Leukocyte Esterase, urine")
OR
(lab_name="pH")
OR
(lab_name="Lactate, ISTAT")
OR
(lab_name="HCO3 (v), ISTAT")
OR
(lab_name="PCO2 (v), ISTAT")
OR
(lab_name="PH (v), ISTAT")
OR
(lab_name="TCO2 (v), ISTAT")
OR
(lab_name="PO2 (v), ISTAT")
OR
(lab_name="O2 Saturation, ISTAT (Ven)")
OR
(lab_name="pCO2 (v)")
OR
(lab_name="pO2 (v)")
OR
(lab_name="pH (v)")
OR
(lab_name="O2 Saturation (v)")
OR
(lab_name="Fibrinogen")
OR
(lab_name="LDH, Total, Ser/Plas")
OR
(lab_name="pCO2 (a)")
OR
(lab_name="pO2 (a)")
OR
(lab_name="pH (a)")
OR
(lab_name="O2 Saturation (a)")
OR
(lab_name="ctO2 (a)")
OR
(lab_name="Hemoglobin A1c")
OR
(lab_name="D-Dimer")
OR
(lab_name="Tacrolimus (FK506)")
OR
(lab_name="Thrombin Time")
OR
(lab_name="Urea Nitrogen,Ser/Plas")
OR
(lab_name="Triglyceride, Ser/Plas")
OR
(lab_name="Potassium, ISTAT")
OR
(lab_name="Hgb(Calc), ISTAT")
OR
(lab_name="Sodium, ISTAT")
OR
(lab_name="Calcium,Ion, ISTAT")
OR
(lab_name="Hct, ISTAT")
OR
(lab_name="Glucose,ISTAT")
OR
(lab_name="Creatinine,ISTAT")
OR
(lab_name="Chloride, ISTAT")
OR
(lab_name="BUN, ISTAT")
OR
(lab_name="Anion Gap, ISTAT")
OR
(lab_name="TCO2, ISTAT")
OR
(lab_name="INR, ISTAT")
OR
(lab_name="PT, ISTAT")
OR
(lab_name="Hct(Calc), ISTAT")
OR
(lab_name="PH, ISTAT")
OR
(lab_name="PCO2, ISTAT")
OR
(lab_name="HCO3, ISTAT")
OR
(lab_name="PO2, ISTAT")
OR
(lab_name="O2 Saturation, ISTAT (Oth)")
OR
(lab_name="Nitrite, urine")
OR
(lab_name="Ketone, urine")
OR
(lab_name="Protein, urine")
OR
(lab_name="Clarity, urine")
OR
(lab_name="Glucose, urine")
OR
(lab_name="Blood, urine")
OR
(lab_name="Glucose, Whole Blood")
OR
(lab_name="Potassium, Whole Bld")
OR
(lab_name="Chloride, Whole Bld")
OR
(lab_name="Sodium, Whole Blood")
OR
(lab_name="NEUT, %")
OR
(lab_name="LYM, %")
OR
(lab_name="MONO, %")
OR
(lab_name="BASO, %")
OR
(lab_name="BASOS, ABS")
OR
(lab_name="EOS, ABS")
OR
(lab_name="EOS, %")
OR
(lab_name="LYM, ABS")
OR
(lab_name="MONO, ABS")
OR
(lab_name="NEUT, ABS")
OR
(lab_name="Basophils")
)