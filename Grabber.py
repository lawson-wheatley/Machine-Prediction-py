import praw as pw

r = pw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',client_id='MAjOkXxMMdqPtw', client_secret="YbCYcplPrWpWnGQvyti8KQpTODc",)
user = r.redditor('Poem_for_your_sprog')
f=open("Data.txt","w")
for comment in user.comments.top():
	try:
		print(comment.body)
		f.write(comment.body+"\n")
	except UnicodeEncodeError:
		print("Unicode found!")
