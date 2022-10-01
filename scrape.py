from urllib3 import urlopen
html = urlopen('http://pythonscraping.com/pages/page1.html')
print(html.read())
