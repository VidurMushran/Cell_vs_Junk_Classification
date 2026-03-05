import pandas as pd
import psycopg2 as pg
import csv

# Locations
base_path = '/mnt/deepstore/Vidur/Junk Classification'

baseline_query = "SELECT oh.slide_id, oh.frame_id, oh.cell_id, oh.x, oh.y, oh.type FROM ocular_hitlist oh JOIN slide on slide.slide_id = split_part(oh.slide_id, '_',1) JOIN staining_batch sb on sb.staining_batch_id = slide.staining_batch_id JOIN protocol p on p.protocol_id = sb.protocol_id WHERE p.name = 'Baseline' ORDER BY oh.slide_id, oh.frame_id, oh.cell_id"

# Connects to our database
def qcon(PGHOST = 'csi-db.usc.edu', PGDB = 'test_msg', PGUSER = 'reader', PGPWD = 'Meta$ta$i$20!7'):
	con = pg.connect(host=PGHOST, dbname=PGDB, user=PGUSER, password=PGPWD)
	print('connected')
	cur = con.cursor()
	return con, cur

# Queries our database
def query(cur, query):
	cur.execute(query)
	records = cur.fetchall()
	col_names = [i[0] for i in cur.description]
	return col_names, records

# Closes connection to our database
def qclose(con, cur):
	cur.close()
	con.close()
	print('closed')

# Stores query as csv
def store_query(file_name, col_names, records):
	fp = open(file_name, 'w')
	f =  csv.writer(fp)
	f.writerow(col_names)
	f.writerows(records)
	fp.close()
	print('stored')
	return file_name

con, cur = qcon()

print('querying cells')
baseline_cols, baseline_data = query(cur, baseline_query)
baseline_df = pd.DataFrame(baseline_data, columns=baseline_cols)

qclose(con, cur)

baseline_df.to_csv(base_path + '/data/baseline/baseline_IDs.csv')
print('Saved')