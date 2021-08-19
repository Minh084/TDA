# output: traige_cohort_labs_basename_filtered

select labs.anon_id, labs.pat_enc_csn_id_coded, order_id_coded, lab_name,
      base_name, ord_num_value, reference_low, reference_high, 
      reference_unit, result_in_range_yn, result_flag, 
      result_time_utc, order_time_utc , taken_time_utc,
from `som-nero-phi-jonc101.shc_core.lab_result` as labs
right join `som-nero-phi-jonc101.triageTD.1_2_cohort` as cohort -- # join labs to cohort
on labs.pat_enc_csn_id_coded = cohort.pat_enc_csn_id_coded
and labs.anon_id = cohort.anon_id
where admit_time >= result_time_utc -- # only labs before admit time
and extract(year from admit_time) > 2014 -- # only CSNs after 2014
and -- the following lab_names and base_names
(
    lab_name in 
        ("Glucose, Non-fasting", "Glucose", "Base Excess Arterial for POC", "Bicarbonate, Art for POC", "Hct (Est)", "HCT, POC", "Hematocrit (Manual Entry) See EMR for details", "Hgb, calculated, POC", "HgB", "Potassium, whole blood, ePOC", "Potassium", "Lactic Acid", "Oxygen Saturation for POC", "Arterial pCO2 for POC", "pH by Meter", "Arterial pH for POC", "Platelets", "Arterial pO2 for POC", "Total Bilirubin", "TCO2, (ISTAT)", "CO2 Arterial Total for POC", "Troponin I, POCT", "WBC count", 
        "Glucose by Meter", "Sodium, Ser/Plas", "Potassium, Ser/Plas", "Calcium, Ser/Plas", "Creatinine, Ser/Plas", "Chloride, Ser/Plas", "CO2, Ser/Plas", "Anion Gap", "Glucose, Ser/Plas", "BUN, Ser/Plas", "Platelet count", "Hematocrit", "Hemoglobin", "WBC", "RBC", "MCV", "MCHC", "MCH", "RDW", "Magnesium, Ser/Plas", "Albumin, Ser/Plas", "Protein, Total, Ser/Plas", "AST (SGOT), Ser/Plas", "ALT (SGPT), Ser/Plas", "Alk P'TASE, Total, Ser/Plas", "Total Bilirubin, Ser/Plas", "Neutrophil, Absolute", "Basophil %", "Eosinophil %", "Neutrophil %", "Monocyte %", "Lymphocyte %", "Monocyte, Absolute", "Lymphocyte, Absolute", "Basophil, Absolute", "Eosinophil, Absolute", "Globulin", "INR", "Prothrombin Time", "Phosphorus, Ser/Plas", "Base Excess, ISTAT", "Lactate, Whole Bld", "TCO2 (a), ISTAT", "pCO2 (a), ISTAT", "PH (a), ISTAT", "PO2 (a), ISTAT", "HCO3 (a), ISTAT", "O2 Saturation, ISTAT", "tCO2", "HCO3", "WBC, urine", "RBC, urine", "TROPONIN I", "Base Excess (vt)", "pH, urine", "Leukocyte Esterase, urine", "pH", "Lactate, ISTAT", "HCO3 (v), ISTAT", "PCO2 (v), ISTAT", "PH (v), ISTAT", "TCO2 (v), ISTAT", "PO2 (v), ISTAT", "O2 Saturation, ISTAT (Ven)", "pCO2 (v)", "pO2 (v)", "pH (v)", "O2 Saturation (v)", "Fibrinogen", "LDH, Total, Ser/Plas", "pCO2 (a)", "pO2 (a)", "pH (a)", "O2 Saturation (a)", "ctO2 (a)", "Hemoglobin A1c", "D-Dimer", "Tacrolimus (FK506)", "Thrombin Time", "Urea Nitrogen,Ser/Plas", "Triglyceride, Ser/Plas", "Potassium, ISTAT", "Hgb(Calc), ISTAT", "Sodium, ISTAT", "Calcium,Ion, ISTAT", "Hct, ISTAT", "Glucose,ISTAT", "Creatinine,ISTAT", "Chloride, ISTAT", "BUN, ISTAT", "Anion Gap, ISTAT", "TCO2, ISTAT", "INR, ISTAT", "PT, ISTAT", "Hct(Calc), ISTAT", "PH, ISTAT", "PCO2, ISTAT", "HCO3, ISTAT", "PO2, ISTAT", "O2 Saturation, ISTAT (Oth)", "Nitrite, urine", "Ketone, urine", "Protein, urine", "Clarity, urine", "Glucose, urine", "Blood, urine", "Glucose, Whole Blood", "Potassium, Whole Bld", "Chloride, Whole Bld", "Sodium, Whole Blood", "NEUT, %", "LYM, %", "MONO, %", "BASO, %", "BASOS, ABS", "EOS, ABS", "EOS, %", "LYM, ABS", "MONO, ABS", "NEUT, ABS", "Basophils")

        or base_name in ('AG', 'AGAP', 'ALB', 'ALT', 'AST', 'ALKP', 'BASO', 'BE', 'BUN', 'CA', 'CAION',  'CL','CO2','CR', 'EOS', 'GLOB', 'GLU', 'GLUURN', 'HCO3A', 'HCO3V', 'HCT', 'HGB', 'INR', 'K', 'LAC', 'LACWBL', 'LYM', 'MG', 'MONO', 'NEUT', 'O2SATA', 'O2SATV', 'PCAGP', 'PCBUN', 'PCCL', 'PCO2A', 'PCO2V', 'PH', 'PHA', 'PHCAI', 'PHV', 'PLT', 'PO2A', 'PO2V', 'PT', 'RBC', 'TBIL', 'TCO2A', 'TCO2V', 'TNI', 'TP', 'WBC', 'XLEUKEST', 'XUKET', 'NA')
)
 
-- or (
-- (lab_name="Glucose by Meter")
-- OR
-- (lab_name="Sodium, Ser/Plas")
-- OR
-- (lab_name="Potassium, Ser/Plas")
-- OR
-- (lab_name="Calcium, Ser/Plas")
-- OR
-- (lab_name="Creatinine, Ser/Plas")
-- OR
-- (lab_name="Chloride, Ser/Plas")
-- OR
-- (lab_name="CO2, Ser/Plas")
-- OR
-- (lab_name="Anion Gap")
-- OR
-- (lab_name="Glucose, Ser/Plas")
-- OR
-- (lab_name="BUN, Ser/Plas")
-- OR
-- (lab_name="Platelet count")
-- OR
-- (lab_name="Hematocrit")
-- OR
-- (lab_name="Hemoglobin")
-- OR
-- (lab_name="WBC")
-- OR
-- (lab_name="RBC")
-- OR
-- (lab_name="Magnesium, Ser/Plas")
-- OR
-- (lab_name="Albumin, Ser/Plas")
-- OR
-- (lab_name="Protein, Total, Ser/Plas")
-- OR
-- (lab_name="AST (SGOT), Ser/Plas")
-- OR
-- (lab_name="ALT (SGPT), Ser/Plas")
-- OR
-- (lab_name="Alk P'TASE, Total, Ser/Plas")
-- OR
-- (lab_name="Total Bilirubin, Ser/Plas")
-- OR
-- (lab_name="Basophil %")
-- OR
-- (lab_name="Eosinophil %")
-- OR
-- (lab_name="Neutrophil %")
-- OR
-- (lab_name="Monocyte %")
-- OR
-- (lab_name="Lymphocyte %")
-- OR
-- (lab_name="Globulin")
-- OR
-- (lab_name="INR")
-- OR
-- (lab_name="Prothrombin Time")
-- OR
-- (lab_name="Phosphorus, Ser/Plas")
-- OR
-- (lab_name="Base Excess, ISTAT")
-- OR
-- (lab_name="Lactate, Whole Bld")
-- OR
-- (lab_name="TCO2 (a), ISTAT")
-- OR
-- (lab_name="pCO2 (a), ISTAT")
-- OR
-- (lab_name="PH (a), ISTAT")
-- OR
-- (lab_name="PO2 (a), ISTAT")
-- OR
-- (lab_name="HCO3 (a), ISTAT")
-- OR
-- (lab_name="O2 Saturation, ISTAT")
-- OR
-- (lab_name="tCO2")
-- OR
-- (lab_name="HCO3")
-- OR
-- (lab_name="WBC, urine")
-- OR
-- (lab_name="TROPONIN I")
-- OR
-- (lab_name="Base Excess (vt)")
-- OR
-- (lab_name="Leukocyte Esterase, urine")
-- OR
-- (lab_name="pH")
-- OR
-- (lab_name="Lactate, ISTAT")
-- OR
-- (lab_name="HCO3 (v), ISTAT")
-- OR
-- (lab_name="PCO2 (v), ISTAT")
-- OR
-- (lab_name="PH (v), ISTAT")
-- OR
-- (lab_name="TCO2 (v), ISTAT")
-- OR
-- (lab_name="PO2 (v), ISTAT")
-- OR
-- (lab_name="O2 Saturation, ISTAT (Ven)")
-- OR
-- (lab_name="pCO2 (v)")
-- OR
-- (lab_name="pO2 (v)")
-- OR
-- (lab_name="pH (v)")
-- OR
-- (lab_name="O2 Saturation (v)")
-- OR
-- (lab_name="pCO2 (a)")
-- OR
-- (lab_name="pO2 (a)")
-- OR
-- (lab_name="pH (a)")
-- OR
-- (lab_name="O2 Saturation (a)")
-- OR
-- (lab_name="ctO2 (a)")
-- OR
-- (lab_name="Hemoglobin A1c")
-- OR
-- (lab_name="Thrombin Time")
-- OR
-- (lab_name="Urea Nitrogen,Ser/Plas")
-- OR
-- (lab_name="Triglyceride, Ser/Plas")
-- OR
-- (lab_name="Potassium, ISTAT")
-- OR
-- (lab_name="Hgb(Calc), ISTAT")
-- OR
-- (lab_name="Sodium, ISTAT")
-- OR
-- (lab_name="Calcium,Ion, ISTAT")
-- OR
-- (lab_name="Hct, ISTAT")
-- OR
-- (lab_name="Glucose,ISTAT")
-- OR
-- (lab_name="Creatinine,ISTAT")
-- OR
-- (lab_name="Chloride, ISTAT")
-- OR
-- (lab_name="BUN, ISTAT")
-- OR
-- (lab_name="Anion Gap, ISTAT")
-- OR
-- (lab_name="TCO2, ISTAT")
-- OR
-- (lab_name="INR, ISTAT")
-- OR
-- (lab_name="PT, ISTAT")
-- OR
-- (lab_name="Hct(Calc), ISTAT")
-- OR
-- (lab_name="PH, ISTAT")
-- OR
-- (lab_name="PCO2, ISTAT")
-- OR
-- (lab_name="HCO3, ISTAT")
-- OR
-- (lab_name="PO2, ISTAT")
-- OR
-- (lab_name="O2 Saturation, ISTAT (Oth)")
-- OR
-- (lab_name="Ketone, urine")
-- OR
-- (lab_name="Glucose, Whole Blood")
-- OR
-- (lab_name="Potassium, Whole Bld")
-- OR
-- (lab_name="Chloride, Whole Bld")
-- OR
-- (lab_name="Sodium, Whole Blood")
-- OR
-- (lab_name="NEUT, %")
-- OR
-- (lab_name="LYM, %")
-- OR
-- (lab_name="MONO, %")
-- OR
-- (lab_name="BASO, %")
-- OR
-- (lab_name="EOS, %")
-- OR
-- (lab_name="Basophils")
-- ))

-- and labs.pat_enc_csn_id_coded in
-- (select distinct(cohort.pat_enc_csn_id_coded) from triage.cohort as cohort)
