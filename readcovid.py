# import pandas as pd

# # Đọc file CSV từ thư mục data/
# df = pd.read_csv('data/countries-aggregated.csv')

# # Hiển thị 5 dòng đầu tiên
# print("5 dòng đầu tiên:")
# print(df.head())

# # Lọc dữ liệu Việt Nam
# vn = df[df['Country'] == 'Vietnam']

# # Chuyển đổi kiểu dữ liệu của cột ngày
# vn['Date'] = pd.to_datetime(vn['Date'])

# # Hiển thị dữ liệu Việt Nam
# print("\nDữ liệu Việt Nam:")
# print(vn.head())


import pandas as pd

df = pd.read_csv('data/countries-aggregated.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("Ngày bắt đầu:", df['Date'].min())
print("Ngày kết thúc:", df['Date'].max())
