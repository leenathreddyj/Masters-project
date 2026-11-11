import json,csv
S=open('../site2/index.html').read().split('\n')
def L(a,b): return '\n'.join(S[a-1:b])   # 1-indexed inclusive
rows=[]
for r in csv.reader(open('../dl/Datasets/pima-indians-diabetes.data.csv')):
    if r: rows.append([float(x) if '.' in x else int(x) for x in r])
assert len(rows)==768 and all(len(r)==9 for r in rows)
eng=open('engine.js').read().replace("if(typeof module!=='undefined')module.exports=engineFactory;","")
css=L(6,226)
head=S[0]+'\n'+S[1]+'\n'+S[2]+'\n'+S[3]+'\n<style>\n'+css+'\n'+open('extra.css').read()+'\n</style>\n'
body=open('body_top.html').read()+L(346,369)+'\n'+L(380,401)+'\n'+open('body_find.html').read()
js='<script>\n'+L(454,461)+'\n'+eng+'\nvar ROWS='+json.dumps(rows,separators=(',',':'))+';\n'+L(463,486)+'\n'+L(488,523)+'\n'+L(689,761)+'\n'+L(763,798)+'\n'+open('app.js').read()+'\n})();\n</script>\n'
open('index.html','w').write(head+body+js)
print(len(head+body+js))
