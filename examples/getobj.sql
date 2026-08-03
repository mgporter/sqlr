with

src as (
  select
    get_ignore_case(data, 'id') as id,
    get_ignore_case(data, 'name') as name,
    get_ignore_case(data, 'age') as age,
    row_id as row_id,
    modified_at as modified_at
  from mydatabase.myschema.persondata
  where modified_at > '2023-01-01 00:00:00'::timestamp_ntz(6)
),

governance as (

  select
    *,
    id as governance_col
  from src
  where id is not null

)

select * from governance