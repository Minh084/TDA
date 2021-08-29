select labs.anon_id, labs.pat_enc_csn_id_coded, order_id_coded, lab_name,
      base_name, reference_low, reference_high, reference_unit
      
from `som-nero-phi-jonc101.shc_core.lab_result` as labs
right join `som-nero-phi-jonc101.triageTD.1_2_cohort` as cohort 

on labs.pat_enc_csn_id_coded = cohort.pat_enc_csn_id_coded
and labs.anon_id = cohort.anon_id

where admit_time >= result_time_utc -- # only labs before admit time

and extract(year from admit_time) > 2014 -- # only CSNs after 2014

and 
(
    lab_name in -- these are the final labs used in Triage project
("Platelet count", "Total Bilirubin, Ser/Plas", "Total Bilirubin", "TROPONIN I", "Troponin I, POCT", "WBC", "WBC count", "Hematocrit", "Hct, ISTAT", "Hct(Calc), ISTAT", "Hct (Est)", "HCT, POC", "Hematocrit (Manual Entry) See EMR for details", "Hemoglobin", "Hgb(Calc), ISTAT", "Hgb, calculated, POC", "HgB", "Potassium, Ser/Plas", "Potassium, ISTAT",  "Potassium, Whole Bld", "Potassium, whole blood, ePOC", "Sodium, Ser/Plas", "Sodium, ISTAT",  "Sodium, Whole Blood", "Chloride, Ser/Plas",  "Chloride, ISTAT",  "Chloride, Whole Bld", "Creatinine, Ser/Plas", "Creatinine,ISTAT", "Anion Gap", "Anion Gap, ISTAT", "Glucose, Ser/Plas", "Glucose,ISTAT", "Glucose, Whole Blood", "Glucose by Meter", "Glucose, Non-fasting", "BUN, Ser/Plas", "BUN, ISTAT", "Urea Nitrogen,Ser/Plas", "Neutrophil, Absolute", "NEUT, ABS", "Neut, ABS (Seg+Band) (man diff)", "Neutrophils, Absolute (Manual Diff)", "Neut, ABS (Seg+Band) (man diff)", "Basophil, Absolute", "BASOS, ABS", "Basophils, ABS (man diff)", "Baso, ABS (man diff)", "Eosinophil, Absolute", "EOS, ABS", "Eosinophils, ABS (man diff)", "Eos, ABS (man diff)", "Lymphocyte, Absolute", "LYM, ABS", "Lymphocytes, ABS (man diff)", "Lym, ABS (man diff)", "Lymphocytes, Abs.", "Monocyte, Absolute", "MONO, ABS", "Monocytes, ABS (man diff)", "Mono, ABS (man diff)", "Lactate, ISTAT", "Lactate, Whole Bld", "Lactic Acid", "HCO3", "HCO3 (a), ISTAT", "Bicarbonate, Art for POC", "HCO3 (v), ISTAT", "HCO3, ISTAT", "O2 Saturation (a)", "O2 Saturation, ISTAT", "Oxygen Saturation for POC", "O2 Saturation (v)",  "O2 Saturation, ISTAT (Ven)", "pCO2 (a)", "pCO2 (a), ISTAT",  "PCO2, ISTAT", "Arterial pCO2 for POC", "pCO2 (v)", "PCO2 (v), ISTAT", "pH (a)", "PH (a), ISTAT", "pH by Meter", "Arterial pH for POC", "pH (v)", "PH (v), ISTAT", "PH, ISTAT", "pO2 (a)", "PO2 (a), ISTAT",  "PO2, ISTAT", "Arterial pO2 for POC", "pO2 (v)", "PO2 (v), ISTAT", "tCO2", "TCO2 (a), ISTAT","TCO2, ISTAT", "TCO2, (ISTAT)", "CO2 Arterial Total for POC", "INR", "INR, ISTAT", "Prothrombin Time", "PT, ISTAT"
)
    or base_name in -- including those added not in the above lab_name list and the rest are from the base_name of the above lab_name list
    ('XLEUKEST', 'EGFR', 'NEUTABS', 'ALB', 'ALKP', 'BE', 'CA', 'CO2', 'GLOB', 'MG', 'MCH', 'PHOS', 'TP', -- not in the lab_name list
     'AG', 'AGAP', 'BASOAB', 'BUN', 'CL', 'CR', 'EGFR', 'EOSAB', 'GLU', 'HCO3', 'HCO3A', 'HCO3V', 'HCT', 'HGB', 'INR', 'K', 'LAC', 'LACWBL', 'LYMAB', 'MONOAB', 'NEUTAB', 'NEUTABS', 'O2SATA', 'O2SATV', 'PCAGP', 'PCBUN', 'PCCL', 'PCO2A', 'PCO2V', 'PH', 'PHA', 'PHV', 'PLT', 'PO2A', 'PO2V', 'PT', 'TBIL', 'TCO2A', 'TNI', 'WBC', 'XLEUKEST', 'NA') -- Neut, ABS (Seg+Band) (man diff) has 2 base name: NEUTAB & NEUTABS
)