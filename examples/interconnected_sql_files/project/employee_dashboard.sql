select
  e.employee_id,
  e.full_name,
  e.email,
  d.department_name,
  d.cost_center,
  e.salary,
  e.annual_salary,
  e.tenure_years,
  coalesce(a.full_address, 'UNKNOWN') as full_address,
  s.daily_total,
  s.order_count,
  s.avg_order_amount,
  s.running_total,
  s.distinct_customer_count,
  round(e.bonus / nullif(e.salary, 0) * 100, 2) as bonus_pct_of_salary
from employee e
left join department d
  on e.department_id = d.department_id
left join address a
  on e.employee_id = a.person_id
left join sales s
  on e.employee_id = s.employee_id
where e.tenure_years >= 1
