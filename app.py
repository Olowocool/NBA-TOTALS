import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="NBA Totals Lab", page_icon="🏀", layout="wide")
st.title("🏀 NBA Totals Lab")
st.caption("Gradient boosting • walk-forward backtest • tonight's totals")

DATE_COL, TARGET = "GAME\xa0DATE", "Point Total"
MIN_EDGE_DEFAULT, MIN_TRAIN_DEFAULT, RETRAIN_DEFAULT = 3.0, 150, 40
MIN_BETS_DEFAULT, ALPHA_DEFAULT, DEFAULT_ODDS = 300, 0.05, -110
FEATURES = [
    'H_WIN%', 'H_PTS_PG', 'H_FGM_PG', 'H_FGA_PG', 'H_CUM_FG%', 'H_3PM_PG', 'H_3PA_PG',
    'H_CUM_3P%', 'H_FTM_PG', 'H_FTA_PG', 'H_CUM_FT%', 'H_OREB_PG', 'H_DREB_PG', 'H_REB_PG',
    'H_AST_PG', 'H_TOV_PG', 'H_STL_PG', 'H_BLK_PG', 'H_BLKA_PG', 'H_PF_PG', 'H_PFD_PG', 'H_+/-_PG',
    'A_WIN%', 'A_PTS_PG', 'A_FGM_PG', 'A_FGA_PG', 'A_CUM_FG%', 'A_3PM_PG', 'A_3PA_PG', 'A_CUM_3P%',
    'A_FTM_PG', 'A_FTA_PG', 'A_CUM_FT%', 'A_OREB_PG', 'A_DREB_PG', 'A_REB_PG', 'A_AST_PG', 'A_TOV_PG',
    'A_STL_PG', 'A_BLK_PG', 'H_BLKA_PG2', 'A_PF_PG', 'A_PFD_PG', 'A_+/-_PG'
]

def new_model():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=.03, max_depth=2,
                                     subsample=.7, min_samples_leaf=5, random_state=42)

def win_profit(odds):
    return 100 / -odds if odds < 0 else odds / 100

def breakeven(odds):
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)

def binom_p_ge(k, n, p):
    # Exact upper-tail binomial probability via recurrence.
    if k <= 0: return 1.
    if k > n or p <= 0: return 0.
    if p >= 1: return 1.
    term = (1-p)**n
    total = 0.
    for i in range(n+1):
        if i >= k: total += term
        if i < n: term *= ((n-i)/(i+1))*(p/(1-p))
    return min(1., total)

def walk_forward(df, min_train, retrain_every):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if len(df) <= min_train:
        raise ValueError(f"Need > {min_train} rows; got {len(df)}.")
    preds, model = np.full(len(df), np.nan), None
    for i in range(min_train, len(df)):
        if model is None or (i-min_train) % retrain_every == 0:
            model = new_model().fit(df.loc[:i-1, FEATURES], df.loc[:i-1, TARGET])
        preds[i] = model.predict(df.loc[[i], FEATURES])[0]
    return df.assign(pred=preds).dropna(subset=["pred"])

def evaluate(d, edge, bootstrap_n=1000):
    b = d[(d.pred-d.OU_Line).abs() >= max(edge, 1e-9)].copy()
    b = b[b[TARGET] != b.OU_Line]
    if b.empty: return None
    over = (b.pred > b.OU_Line).to_numpy()
    win = np.where(over, b[TARGET] > b.OU_Line, b[TARGET] < b.OU_Line)
    odds = np.where(over, b.Over_Odds, b.Under_Odds).astype(float)
    profits = np.array([win_profit(o) if w else -1 for w,o in zip(win,odds)])
    n, wins = len(b), int(win.sum())
    be = float(np.mean([breakeven(o) for o in odds]))
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(profits, n, replace=True).mean() for _ in range(bootstrap_n)])
    result = dict(edge=edge, bets=n, wins=wins, hit=wins/n, breakeven=be, roi=profits.mean(),
                  roi_lo=np.percentile(boot,2.5), roi_hi=np.percentile(boot,97.5),
                  p_value=binom_p_ge(wins,n,be))
    if "OU_Close" in b:
        result["CLV_pts"] = float(np.where(over,b.OU_Close-b.OU_Line,b.OU_Line-b.OU_Close).mean())
    return result

def make_demo(n=900):
    rng=np.random.default_rng(1)
    X=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    mean=224+3*X.iloc[:,1]+2*X.iloc[:,23]+rng.normal(0,3,n)
    d=X.copy(); d[TARGET]=np.round(mean+rng.normal(0,14,n)).astype(int)
    d["Game_Id"]=np.arange(n); d[DATE_COL]=pd.date_range("2018-10-20",periods=n,freq="D")
    lines=pd.DataFrame({"Game_Id":d.Game_Id,"OU_Line":np.round((mean+rng.normal(0,2,n))*2)/2})
    tonight=X.tail(5).reset_index(drop=True); tonight.insert(0,"AwayTeam",list("VWXYZ")); tonight.insert(0,"HomeTeam",list("ABCDE"))
    tonight["OU_Line"]=[224.5,226.5,223.5,225.5,222.5]
    return d,lines,tonight

def csv_upload(label, key):
    f=st.file_uploader(label,type=["csv"],key=key)
    if f is None:return None
    return pd.read_csv(f)

with st.sidebar:
    st.header("Model controls")
    min_edge=st.number_input("Minimum edge (points)",0.0,15.0,MIN_EDGE_DEFAULT,.5)
    min_train=st.number_input("Initial training games",50,2000,MIN_TRAIN_DEFAULT,25)
    retrain=st.number_input("Retrain every N games",1,500,RETRAIN_DEFAULT,5)
    min_bets=st.number_input("Minimum bets for validation",1,5000,MIN_BETS_DEFAULT,50)
    alpha=st.number_input("Significance level",.001,.2,ALPHA_DEFAULT,.005,format="%.3f")
    boots=st.select_slider("Bootstrap samples",options=[250,500,1000,2000],value=1000)
    st.divider()
    demo=st.button("Load synthetic demo data",use_container_width=True)

if demo:
    d,lines,tonight=make_demo()
    st.session_state["train_demo"]=d;st.session_state["lines_demo"]=lines;st.session_state["tonight_demo"]=tonight

tabs=st.tabs(["📊 Backtest & Verdict","🌙 Tonight's Picks","📁 Data & Setup"])
with tabs[2]:
    st.subheader("Upload your CSV files")
    st.write("Training file: Game_Id, GAME DATE, Point Total, and all 44 model features.")
    train_file=st.file_uploader("Historical training CSV",type="csv",key="train_file")
    lines_file=st.file_uploader("Historical lines CSV (Game_Id, OU_Line; optional OU_Close, odds)",type="csv",key="lines_file")
    tonight_file=st.file_uploader("Tonight games CSV (HomeTeam, AwayTeam, 44 features; optional OU_Line)",type="csv",key="tonight_file")
    st.info("American odds are accepted. Missing Over/Under odds default to -110. Keep all features pregame-only to prevent leakage.")
    st.code("pip install streamlit pandas numpy scikit-learn\\nstreamlit run app.py",language="bash")

def get_data():
    train = pd.read_csv(st.session_state["train_file"]) if st.session_state.get("train_file") else st.session_state.get("train_demo")
    lines = pd.read_csv(st.session_state["lines_file"]) if st.session_state.get("lines_file") else st.session_state.get("lines_demo")
    tonight = pd.read_csv(st.session_state["tonight_file"]) if st.session_state.get("tonight_file") else st.session_state.get("tonight_demo")
    if train is not None:
        date_col=DATE_COL if DATE_COL in train else DATE_COL.replace("\xa0"," ")
        if date_col != DATE_COL: train=train.rename(columns={date_col:DATE_COL})
    return train,lines,tonight

train,lines,tonight=get_data()
required=["Game_Id",DATE_COL,TARGET]+FEATURES
if train is not None:
    missing=[c for c in required if c not in train]
    if missing:
        st.error(f"Training CSV missing columns: {missing}")
        train=None
    else:
        train=train.drop_duplicates("Game_Id").dropna(subset=FEATURES+[TARGET]).copy()
        train[DATE_COL]=pd.to_datetime(train[DATE_COL],errors="coerce")
        train=train.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
if lines is not None:
    if not {"Game_Id","OU_Line"}.issubset(lines.columns):
        st.error("Lines CSV must contain Game_Id and OU_Line.");lines=None
    else:
        lines=lines.copy()
        for col in ["Over_Odds","Under_Odds"]:
            if col not in lines:lines[col]=DEFAULT_ODDS
            lines[col]=pd.to_numeric(lines[col],errors="coerce").fillna(DEFAULT_ODDS)

with tabs[0]:
    st.subheader("Walk-forward profitability test")
    if train is None or lines is None:
        st.warning("Upload both training data and historical lines, or click 'Load synthetic demo data' in the sidebar.")
    else:
        if st.button("Run walk-forward backtest",type="primary"):
            try:
                with st.spinner("Training walk-forward models and evaluating bets…"):
                    wf=walk_forward(train,int(min_train),int(retrain))
                    joined=wf.merge(lines,on="Game_Id",how="inner",validate="many_to_one")
                    joined=joined.dropna(subset=["OU_Line"])
                    if joined.empty:raise ValueError("No matching historical lines for out-of-sample Game_Id values.")
                    mae_model=(joined[TARGET]-joined.pred).abs().mean()
                    mae_line=(joined[TARGET]-joined.OU_Line).abs().mean()
                    result=pd.DataFrame([x for x in [evaluate(joined,e,boots) for e in sorted(set([0,1,2,3,4,5,6,float(min_edge)]))] if x])
                    st.session_state["backtest_result"]=result
                    st.session_state["backtest_joined"]=joined
                    st.session_state["mae_model"]=mae_model;st.session_state["mae_line"]=mae_line
            except Exception as ex:st.error(str(ex))
        if "backtest_result" in st.session_state:
            res=st.session_state["backtest_result"]
            joined=st.session_state["backtest_joined"]
            maem,mael=st.session_state["mae_model"],st.session_state["mae_line"]
            c1,c2,c3=st.columns(3)
            c1.metric("Out-of-sample games",len(joined));c2.metric("Model MAE",f"{maem:.2f}");c3.metric("Line MAE",f"{mael:.2f}")
            st.write("Model has lower MAE than line:",maem<mael)
            st.dataframe(res,use_container_width=True)
            fixed=res[np.isclose(res.edge,float(min_edge))]
            if fixed.empty:
                reasons=["No bets qualified at the fixed minimum edge."]
            else:
                r=fixed.iloc[0];reasons=[]
                if r.bets<min_bets:reasons.append(f"{int(r.bets)} bets below required {min_bets}")
                if r.p_value>=alpha:reasons.append(f"p-value {r.p_value:.4f} ≥ {alpha}")
                if r.roi_lo<=0:reasons.append(f"ROI 95% interval [{r.roi_lo:.1%}, {r.roi_hi:.1%}] includes zero")
                if "CLV_pts" in r and r.CLV_pts<=0:reasons.append(f"CLV {r.CLV_pts:.2f} is not positive")
                if maem>=mael:reasons.append("Model MAE does not beat sportsbook line")
            validated=not reasons if not fixed.empty else False
            st.success("VALIDATED — historical evidence meets all configured checks") if validated else st.error("UNVALIDATED — "+"; ".join(reasons))
            st.download_button("Download backtest CSV",res.to_csv(index=False).encode(), "nba_backtest.csv","text/csv")

with tabs[1]:
    st.subheader("Tonight's predictions")
    if train is None or tonight is None:
        st.info("Upload training and tonight CSVs, or load synthetic demo data.")
    else:
        missing=[x for x in FEATURES+["HomeTeam","AwayTeam"] if x not in tonight]
        if missing:st.error(f"Tonight CSV missing columns: {missing}")
        elif st.button("Fit on all history & predict tonight",type="primary"):
            try:
                model=new_model().fit(train[FEATURES],train[TARGET])
                t=tonight.dropna(subset=FEATURES).copy()
                out=t[["HomeTeam","AwayTeam"]].copy()
                out["Predicted total"]=model.predict(t[FEATURES]).round(1)
                out["Validation"]="VALIDATED" if st.session_state.get("validated",False) else "UNVALIDATED"
                if "OU_Line" in t:
                    out["Line"]=t["OU_Line"]
                    out["Edge pts"]=(out["Predicted total"]-out["Line"]).round(1)
                    out["Pick"]=np.where(out["Edge pts"].abs()<min_edge,"PASS",np.where(out["Edge pts"]>0,"OVER","UNDER"))
                st.session_state["tonight_out"]=out
            except Exception as ex:st.error(str(ex))
        if "tonight_out" in st.session_state:
            st.dataframe(st.session_state["tonight_out"],use_container_width=True)
            st.download_button("Download tonight picks",st.session_state["tonight_out"].to_csv(index=False).encode(),"tonight_picks.csv","text/csv")
