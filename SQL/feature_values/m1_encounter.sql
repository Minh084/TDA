SELECT c.*,
    e.inpatient_data_id_coded, 
    e.enc_type, e.visit_type, e.acuity_level, e.ACUITY_LEVEL_C,
    e.hosp_admsn_time_jittered_utc
FROM 
    `som-nero-phi-jonc101.shc_core.encounter` as e
RIGHT JOIN 
    `som-nero-phi-jonc101.triageTD.cohort0` as c
ON (c.anon_id=e.anon_id and c.pat_enc_csn_id_coded=e.pat_enc_csn_id_coded)
ORDER BY
  c.anon_id