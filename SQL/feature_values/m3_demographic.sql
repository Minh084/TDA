SELECT c.anon_id,
    d.gender, d.canonical_race as race, d.language, 
    d.recent_ht_in_cms as recent_height_cm, d.recent_wt_in_kgs as recent_weight_kg,
    d.insurance_payor_name as insurance, d.recent_conf_enc_jittered as recent_date, 
    DATE(CAST(d.birth_date_jittered as TIMESTAMP)) as dob
FROM 
    `som-nero-phi-jonc101.shc_core.demographic` as d
JOIN 
    `som-nero-phi-jonc101.triageTD.cohort0` as c
ON c.anon_id=d.anon_id
ORDER BY
  c.anon_id
