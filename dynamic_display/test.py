# test_pyecharts.py
from pyecharts.charts import Line
from pyecharts import options as opts

line = Line()
line.add_xaxis(["A", "B", "C"])
line.add_yaxis("Series", [1, 3, 2])
line.set_global_opts(title_opts=opts.TitleOpts(title="Test Chart"))
line.render("test_chart.html")

print("测试图表已生成: test_chart.html")