with src as (
  select test_id, name from mydatabase.myschema.table_in_cte
)

select
  id,
  test.name,
  date_trunc('day', modified_at) as modified_at,
  list_extract(titles, 1) as first_title
from mydatabase.myschema.test
inner join src
    on test.id = src.test_id
where modified_at > cast('2023-01-01 00:00:00' as timestamp_ntz(6))
