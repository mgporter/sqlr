with src as (
  select test_id, name from mydatabase.myschema.table_in_cte
)

select
  id,
  table1.name,
  date_trunc('day', modified_at) as modified_at,
  list_extract(titles, 1) as first_title,
  mistyped_tablename.col1 as col1,
  mistyped_tablename.col2.jsonfield as col2,
  mistyped_tablename['col3'] as col3,
  mistyped_tablename.col4['jsonfield'] as col4
from mydatabase.myschema.table1
inner join src
    on id = src.test_id
where modified_at > cast('2023-01-01 00:00:00' as timestamp_ntz(6))
