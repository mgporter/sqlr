with deduped as (
  select
    *,
    row_number() over (partition by employee_id order by modified_at desc) as rn
  from mydatabase.myschema.raw_employee
  where status in ('ACTIVE', 'ON_LEAVE')
),
current_employee as (
  select
    employee_id,
    department_id,
    first_name || ' ' || last_name as full_name,
    lower(email) as email,
    cast(salary as decimal(10, 2)) as salary,
    salary * 12 as annual_salary,
    ifnull(bonus, 0) as bonus,
    hire_date,
    datediff(year, hire_date, current_date()) as tenure_years,
    *
  from deduped
  where rn = 1
    and department_id in (
      select department_id
      from mydatabase.myschema.raw_department
      where is_active = true
    )
)
select * from current_employee
