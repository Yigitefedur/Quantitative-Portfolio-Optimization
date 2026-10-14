# ==============================================================================
# 🧠 EVRENSEL KUANT MOTORU (ZIRHLI SÜRÜM) 🧠
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import nltk
import time
import matplotlib.pyplot as plt
import scipy.optimize as sco
from tefas import Crawler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

nltk.download('vader_lexicon', quiet=True)

# ------------------------------------------------------------------------------
# 🛠️ 1. CÜZDANIN (SADECE BURAYI DEĞİŞTİR) 🛠️
# ------------------------------------------------------------------------------
SERMAYE = 100000  
BASLANGIC_TARIHI = '2024-01-01'
BITIS_TARIHI = '2026-05-01'

BENIM_PORTFOYUM = {
    'SMR': ['SMR'],                # Amerikan NuScale Power Hissesi
    'IPJ': ['TSLA', 'BYDDY'],      # TEFAS Fonu
    'SAS': ['TUR', 'AKBNK.IS']     # TEFAS Fonu
}
# ------------------------------------------------------------------------------

def evrensel_kuant_motoru():
    print("=================================================================")
    print("🌍 EVRENSEL OTO-KEŞİF MOTORU BAŞLATILIYOR... 🌍")
    print("=================================================================\n")
    
    varliklar = list(BENIM_PORTFOYUM.keys())
    
    # --- AŞAMA 1: YAHOO FINANCE TARAMASI ---
    print("📡 1/5 Varlıkların Kökeni Analiz Ediliyor ve Fiyatlar Çekiliyor...")
    df_yf = pd.DataFrame()
    basarili_yf = []
    
    try:
        ham_yf = yf.download(varliklar, start=BASLANGIC_TARIHI, end=BITIS_TARIHI, progress=False)['Close']
        if isinstance(ham_yf, pd.Series) and not ham_yf.empty:
            df_yf = ham_yf.to_frame(name=varliklar[0])
            basarili_yf = [varliklar[0]]
        elif not ham_yf.empty:
            df_yf = ham_yf.dropna(axis=1, how='all')
            basarili_yf = list(df_yf.columns)
            
        if not df_yf.empty:
            df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
    except Exception:
        pass

    for v in basarili_yf:
        print(f"   🟢 {v:<10} -> Yahoo Finance (Küresel Hisse/ETF) olarak algılandı.")

    # --- AŞAMA 2: TEFAS TARAMASI ---
    kalan_varliklar = [v for v in varliklar if v not in basarili_yf]
    df_tefas = pd.DataFrame()
    basarili_tefas = []

    if kalan_varliklar:
        crawler = Crawler()
        tefas_veriler = []
        for fon in kalan_varliklar:
            basarili = False
            deneme = 0
            while not basarili and deneme < 3:
                try:
                    veri = crawler.fetch(start=BASLANGIC_TARIHI, end=BITIS_TARIHI, name=fon, columns=['date', 'code', 'price'])
                    if veri is not None and not veri.empty:
                        tefas_veriler.append(veri)
                        basarili_tefas.append(fon)
                        print(f"   🔵 {fon:<10} -> TEFAS Yatırım Fonu olarak algılandı.")
                        basarili = True
                        time.sleep(1) 
                except:
                    deneme += 1
                    time.sleep(2)

        if tefas_veriler:
            ham_tefas = pd.concat(tefas_veriler, ignore_index=True)
            ham_tefas['price'] = pd.to_numeric(ham_tefas['price'], errors='coerce')
            df_tefas = ham_tefas.pivot(index='date', columns='code', values='price').dropna()
            df_tefas.index = pd.to_datetime(df_tefas.index)

    # --- AŞAMA 3: İKİ DÜNYAYI BİRLEŞTİR (SENKRONİZASYON) ---
    tum_fiyatlar = pd.concat([df_tefas, df_yf], axis=1).fillna(method='ffill').dropna() 
    
    gecerli_varliklar = list(tum_fiyatlar.columns)
    varlik_sayisi = len(gecerli_varliklar)
    
    if varlik_sayisi < 2:
        print("\n❌ HATA: Yeterli geçerli varlık bulunamadı (En az 2 adet gerekiyor).")
        return

    # --- KUANT ZIRHI (SVD ÇÖKMESİNİ ENGELLEYEN İLAÇ) ---
    getiriler = tum_fiyatlar.pct_change()
    getiriler.replace([np.inf, -np.inf], np.nan, inplace=True) # Sonsuzlukları temizle
    getiriler = getiriler.dropna()
    
    yillik_getiri = getiriler.mean() * 252
    cov_matrisi = (getiriler.cov() * 252).values
    
    cov_matrisi = (cov_matrisi + cov_matrisi.T) / 2      # Matrisi simetrik zorla
    cov_matrisi += np.eye(varlik_sayisi) * 1e-8          # SVD için mikroskobik pürüz ekle
    # ----------------------------------------------------
    
    print(f"\n✅ Tüm zaman serileri hizalandı ve zırhlandı. Aktif Portföy: {gecerli_varliklar}\n")

    # --- AŞAMA 4: NLP HABER OKUMA MOTORU ---
    print(f"📰 2/5 {varlik_sayisi} Varlık İçin Küresel Medya Taranıyor...")
    sia = SentimentIntensityAnalyzer()
    dinamik_hedefler = []
    dinamik_guvenler = []
    
    for varlik in gecerli_varliklar:
        hisseler = BENIM_PORTFOYUM.get(varlik, [varlik])
        toplam_skor = 0
        gecerli_haber = 0
        
        for hisse in hisseler:
            haberler = yf.Ticker(hisse).news
            if haberler:
                for haber in haberler[:5]:
                    baslik = haber.get('title', '') or haber.get('content', {}).get('title', '')
                    if baslik:
                        toplam_skor += sia.polarity_scores(baslik)['compound']
                        gecerli_haber += 1
                        
        ortalama_skor = toplam_skor / gecerli_haber if gecerli_haber > 0 else 0
        guven = min(max(abs(ortalama_skor) + 0.10, 0.10), 0.95)
        hedef = 0.30 + (ortalama_skor * 0.30)
        dinamik_hedefler.append(hedef)
        dinamik_guvenler.append(guven)

    # --- AŞAMA 5: BLACK-LITTERMAN & OPTİMİZASYON ---
    print("\n🧮 3/5 Yapay Zeka Görüşleri Optimizasyon Matrislerine İşleniyor...")
    P = np.eye(varlik_sayisi)
    Q = np.array(dinamik_hedefler)
    tau = 0.05
    omega = np.zeros((varlik_sayisi, varlik_sayisi))
    
    for i in range(varlik_sayisi):
        varyans_i = np.dot(np.dot(P[i:i+1, :], tau * cov_matrisi), P[i:i+1, :].T)[0, 0]
        c = dinamik_guvenler[i]
        omega[i, i] = varyans_i * ((1.0 - c) / c)

    inv_tau_cov = np.linalg.inv(tau * cov_matrisi)
    inv_omega = np.linalg.inv(omega)
    sol_kisim = np.linalg.inv(inv_tau_cov + np.dot(np.dot(P.T, inv_omega), P))
    sag_kisim = np.dot(inv_tau_cov, yillik_getiri.values) + np.dot(np.dot(P.T, inv_omega), Q)
    bl_getirisi = np.dot(sol_kisim, sag_kisim)

    def sharpe_maksimize_et(agirliklar):
        getiri = np.sum(bl_getirisi * agirliklar)
        risk = np.sqrt(np.dot(agirliklar.T, np.dot(cov_matrisi, agirliklar)))
        return -(getiri / risk) 

    kisitlar = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    sinirlar = tuple((0, 1) for _ in range(varlik_sayisi))
    baslangic = np.ones(varlik_sayisi) / varlik_sayisi

    opt_sonuc = sco.minimize(sharpe_maksimize_et, baslangic, method='SLSQP', bounds=sinirlar, constraints=kisitlar)
    optimum_agirliklar = np.where(opt_sonuc.x < 0.001, 0, opt_sonuc.x)
    optimum_agirliklar = optimum_agirliklar / np.sum(optimum_agirliklar) 

    # --- AŞAMA 6: MONTE CARLO SİMÜLASYONU ---
    print("\n🔮 4/5 Gelecek 1 Yıl İçin 1000 Paralel Evren Simüle Ediliyor (Risk Analizi)...")
    gun_sayisi = 252             
    sim_sayisi = 1000     
    gunluk_mu = bl_getirisi / 252
    gunluk_cov = cov_matrisi / 252

    sim_portfoy = np.zeros((gun_sayisi, sim_sayisi))
    sim_fonlar = np.zeros((gun_sayisi, sim_sayisi, varlik_sayisi))

    for i in range(sim_sayisi):
        rastgele_getiriler = np.random.multivariate_normal(gunluk_mu, gunluk_cov, gun_sayisi)
        portfoy_getirisi = np.dot(rastgele_getiriler, optimum_agirliklar)
        sim_portfoy[:, i] = SERMAYE * np.cumprod(1 + portfoy_getirisi)
        for j in range(varlik_sayisi):
            sim_fonlar[:, i, j] = SERMAYE * np.cumprod(1 + rastgele_getiriler[:, j])

    portfoy_final = sim_portfoy[-1, :]
    portfoy_medyan = np.percentile(portfoy_final, 50)
    portfoy_var = np.percentile(portfoy_final, 5)

    # --- NİHAİ RAPOR ---
    print("\n=================================================================")
    print("🏆 NİHAİ YATIRIM EMRİ (ELENENLER VE SEÇİLENLER) 🏆")
    print("=================================================================")
    
    secilenler = []
    secilen_agirliklar = []
    secilen_medyanlar = []
    secilen_var = []
    
    for i, varlik in enumerate(gecerli_varliklar):
        agirlik = optimum_agirliklar[i]
        if agirlik > 0:
            print(f"🟢 {varlik:<10} | % {agirlik*100:>5.1f} | {agirlik * SERMAYE:,.0f} TL")
            secilenler.append(varlik)
            secilen_agirliklar.append(agirlik)
            secilen_medyanlar.append(np.percentile(sim_fonlar[-1, :, i], 50))
            secilen_var.append(np.percentile(sim_fonlar[-1, :, i], 5))
        else:
            print(f"🔴 {varlik:<10} | % 0.0   | 0 TL (ELENDİ)")
    
    print("-" * 65)
    print(f"👉 KÜMÜLATİF BEKLENEN KAPANIŞ : {portfoy_medyan:,.0f} TL")
    print(f"🛡️ KÜMÜLATİF KRİZ DİBİ (%5 VaR): {portfoy_var:,.0f} TL")
    print("=================================================================\n")

    # --- GÖRSELLEŞTİRME ---
    print("📊 5/5 Görseller Çiziliyor...")
    if len(secilenler) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pasta Grafiği
        ax1.pie(secilen_agirliklar, labels=secilenler, autopct='%1.1f%%', startangle=140, 
                colors=plt.cm.Set3.colors, wedgeprops={'edgecolor': 'black'})
        ax1.set_title('Otonom Cüzdan Dağılımı', fontsize=14, fontweight='bold')

        # Risk Bar Grafiği
        labels_bar = ['Nihai Portföy'] + secilenler
        med_bar = [portfoy_medyan] + secilen_medyanlar
        var_bar = [portfoy_var] + secilen_var
        x = np.arange(len(labels_bar))
        width = 0.35

        ax2.bar(x - width/2, med_bar, width, label='Beklenen (Medyan)', color='teal')
        ax2.bar(x + width/2, var_bar, width, label='Kriz Anı (%5 VaR)', color='crimson')
        ax2.axhline(y=SERMAYE, color='black', linestyle='--', linewidth=2, label=f'Başlangıç ({SERMAYE:,.0f} TL)')
        ax2.set_ylabel('Sermaye (TL)', fontsize=12)
        ax2.set_title('Monte Carlo Risk/Getiri Kıyaslaması', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_bar, fontsize=10, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Makineyi Çalıştır
evrensel_kuant_motoru()
