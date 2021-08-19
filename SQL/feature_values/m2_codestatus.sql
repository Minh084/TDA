SELECT c.anon_id, c.pat_enc_csn_id_coded, c.admit_time_jittered,
    o.order_type, o.order_status, o.display_name, o.description, 
    o.order_time_jittered_utc
FROM 
    `som-nero-phi-jonc101.shc_core.order_proc` as o
JOIN 
    `som-nero-phi-jonc101.triageTD.1_1_cohort` as c
ON (c.anon_id=o.anon_id and c.pat_enc_csn_id_coded=o.pat_enc_csn_id_coded)
WHERE o.order_type = "Code Status"
ORDER BY
  c.anon_id
