import json
import sqlite3
import bisect
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import config

_db_fields = ("id",)
_table_name = "response"

_sql_create_table = f'''CREATE TABLE IF NOT EXISTS {_table_name} (
    id INTEGER PRIMARY KEY NOT NULL,
    paper_text  TEXT NOT NULL,
    paper_type  TEXT NOT NULL,
    params TEXT NOT NULL,
    output TEXT NOT NULL,
    conf   TEXT NOT NULL,
    label  TEXT NOT NULL,
    pred   TEXT,
    timestamp datetime DEFAULT CURRENT_TIMESTAMP
);
'''

_some_sql_cmds = '''
-- SELECT count(id)
-- FROM response
-- WHERE pred="NoAns";

-- SELECT count(id)
-- FROM response
-- WHERE pred=label;

-- SELECT count(id)
-- FROM response
-- WHERE pred=label AND conf="Arxiv_Kaggle_Data";

-- SELECT count(id)
-- FROM response
-- WHERE pred=label AND conf!="Arxiv_Kaggle_Data";

-- SELECT count(id)
-- FROM response
-- WHERE pred="Accept";

-- SELECT count(id)
-- FROM response
-- WHERE pred="Reject";
'''


@dataclass
class DataEntry:
    paper_text: str
    paper_type: str
    file: str
    conf: str
    label: str
    id: int = -1
    params: List[dict] = field(default_factory=lambda: [])
    output: List[str] = field(default_factory=lambda: [])
    pred: str = ""

    def __str__(self):
        return f"{self.conf} - ({self.paper_type}){self.file} - {self.label}"

    def infer_pred(self):
        if self.pred == "Error":
            return self.pred
        try:
            if isinstance(self.output, list):
                output_lower = self.output[-1].lower()
            elif isinstance(self.output, str):
                output_lower = self.output.lower()
            else:
                raise NotImplementedError
                # assert isinstance(self.output, list) and isinstance(self.output, str)
            flag_accept = "accept" in output_lower
            flag_reject = "reject" in output_lower
            if flag_reject and flag_accept:
                self.pred = "Unknown"
            elif flag_accept:
                self.pred = config.ACCEPT
            elif flag_reject:
                self.pred = config.REJECT
            else:
                self.pred = "NoAns"
        except:
            self.pred = "Error"
        return self.pred


class DBManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.conn.execute(_sql_create_table)
        self.cur = self.conn.cursor()

    def insert(self, data: DataEntry):
        data_to_insert = (data.paper_text,
                          data.paper_type,
                          json.dumps(data.output),
                          data.conf,
                          data.label,
                          data.pred,
                          json.dumps(data.params))
        self.cur.executemany(f'''INSERT INTO {_table_name}
                                 (paper_text, paper_type, output, conf, label, pred, params)
                                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
                             [data_to_insert])
        self.conn.commit()

    def update(self, data: DataEntry):
        data_to_update = (json.dumps(data.output),
                          json.dumps(data.params),
                          data.pred,
                          data.id)
        self.cur.execute(f'''UPDATE {_table_name} 
                                SET output = ?,
                                    params = ?,
                                    pred = ?,
                                    timestamp = CURRENT_TIMESTAMP
                                WHERE id = ?''',
                         data_to_update)
        self.conn.commit()

    def get_err_indices(self, rearrange=False):
        self.cur.execute(f'''SELECT id from {_table_name} where pred="Error" ORDER BY id''')
        rows = self.cur.fetchall()
        indices = [r[0] - 1 for r in rows]
        if rearrange:
            pivot = 1988 // 2
            i_pivot = bisect.bisect(indices, pivot)
            left, right = indices[:i_pivot], indices[i_pivot:]
            print(len(right) - len(left))
            indices_rearranged = right[:len(right) - len(left)]
            right = right[len(right) - len(left):]
            assert len(left) == len(right)
            for il, ir in zip(left, right):
                indices_rearranged.append(il)
                indices_rearranged.append(ir)
            assert sorted(indices_rearranged) == indices
            return indices_rearranged
        return indices

    def update_pred(self):
        pass

    def load_all(self):
        self.cur.execute(f'''SELECT id, paper_text, paper_type, conf, label, params, output, pred
                            FROM {_table_name} ORDER BY id''')
        rows = self.cur.fetchall()
        data_list = []
        for r in rows:
            output = json.loads(r[6]) if r[2] == 'txt' else r[6]
            data_list.append(DataEntry(
                id=r[0],
                paper_text=r[1],
                paper_type=r[2],
                conf=r[3],
                label=r[4],
                params=json.loads(r[5]),
                output=output,
                pred=r[7],
                file="Unknown",
            ))
        return data_list

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = DBManager(db_file=config.GPT_EXP_CFG_TXT.db_file)
    indices = db.get_err_indices(rearrange=True)
