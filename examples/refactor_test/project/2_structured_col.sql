with src as (
  select test_id, name from mydatabase.myschema.table_in_cte
)
select
  id,
  table1.name,
  date_trunc('day', modified_at) as modified_at,
  list_extract(titles, 1) as first_title,
  structured_col['myfield'] as my_field,
from mydatabase.myschema.table1
inner join src
    on id = src.test_id
where modified_at > cast('2023-01-01 00:00:00' as timestamp_ntz(6))
