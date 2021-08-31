SELECT cohort.*,
        order_id_coded, lab_name, base_name, ord_value, ord_num_value, 
        reference_low, reference_high, reference_unit, result_in_range_yn, result_flag, 
        result_time_utc, order_time_utc , taken_time_utc
      
FROM `som-nero-phi-jonc101.shc_core.lab_result` as labs
RIGHT JOIN `som-nero-phi-jonc101.triageTD.1_3_cohort` as cohort  -- # join labs to cohort

ON labs.pat_enc_csn_id_coded = cohort.pat_enc_csn_id_coded
AND labs.anon_id = cohort.anon_id

WHERE admit_time >= result_time_utc  -- # only labs before admit time
AND extract(year from admit_time) > 2014  -- # only CSNs after 2014
AND base_name in 
    ('AG', 'AGAP', 'BASOAB', 'BUN', 'CL', 'CR', 'EGFR', 'EOSAB', 'GLU', 'HCO3', 'HCO3A', 'HCO3V', 
 'HCT', 'HGB', 'INR', 'K', 'LAC', 'LACWBL', 'LYMAB', 'MONOAB', 'NEUTAB', 'NEUTABS', 'O2SATA', 
 'O2SATV', 'PCAGP', 'PCBUN', 'PCCL', 'PCO2A', 'PCO2V', 'PH', 'PHA', 'PHV', 'PLT', 'PO2A', 'PO2V',
 'PT', 'TBIL', 'TCO2A', 'TNI', 'WBC', 'NA', 'ALB', 'ALKP', 'ALT', 'AST', 'BE', 'CA', 'CO2', 
 'GLOB', 'MCH', 'RDW', 'TP') -- 'GLUURN' removed, all NA