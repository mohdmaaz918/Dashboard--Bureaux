const dropZone  = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const dashboard  = document.getElementById('dashboard');
let isProcessing = false;
let _lastData    = null;  // holds the last successful API response for Excel export

// ─── Formatters ────────────────────────────────────────────────
function formatCurrency(val) {
    if (val === null || val === undefined) return "$0";
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setHTML(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}

// ─── Drag & Drop Handlers ──────────────────────────────────────
dropZone.addEventListener('click', () => { if (!isProcessing) fileInput.click(); });

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (!isProcessing) dropZone.style.borderColor = 'var(--accent)';
});

dropZone.addEventListener('dragleave', () => {
    if (!isProcessing) dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (isProcessing) return;
    dropZone.style.borderColor = 'var(--glass-border)';
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (isProcessing) return;
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// ─── File Upload & API call ────────────────────────────────────
async function handleFile(file) {
    if (!file.name.endsWith('.json')) {
        alert("Please upload a valid Equifax JSON file.");
        return;
    }
    isProcessing = true;
    document.getElementById('dropzone-content').innerHTML = `<div class="upload-text"><i class="fas fa-spinner fa-spin"></i> Processing Report...</div>`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error while processing file.");
        }
        const data = await response.json();
        _lastData = data;               // store for Excel export
        renderDashboard(data);
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) exportBtn.style.display = 'flex';
        const uploadNewBtn = document.getElementById('upload-new-btn');
        if (uploadNewBtn) uploadNewBtn.style.display = 'flex';
    } catch (err) {
        alert("Error: " + err.message);
        document.getElementById('dropzone-content').innerHTML = `
            <i class="upload-icon fas fa-cloud-upload-alt"></i>
            <div class="upload-text">Upload Equifax Response JSON</div>
            <div class="upload-hint">Drag &amp; drop or click to browse</div>
        `;
    } finally {
        isProcessing = false;
        fileInput.value = "";
    }
}

// ─── Reset to Upload Screen ────────────────────────────────────
function resetToUpload() {
    _lastData = null;
    fileInput.value = '';
    document.getElementById('dropzone-content').innerHTML = `
        <i class="upload-icon fas fa-cloud-upload-alt"></i>
        <div class="upload-text">Upload Equifax Response JSON</div>
        <div class="upload-hint">Drag &amp; drop or click to browse</div>
    `;
    dashboard.style.display = 'none';
    dropZone.style.display = '';
    document.getElementById('export-btn').style.display = 'none';
    document.getElementById('upload-new-btn').style.display = 'none';
}

// ─── Main Render ───────────────────────────────────────────────
function renderDashboard(data) {
    dropZone.style.display = 'none';
    dashboard.style.display = 'grid';

    const id = data.identity || {};
    const m  = data.metrics  || {};

    // ── IDENTITY ──────────────────────────────────────────────
    setHTML('id-name',       id.full_name || "<span class='text-muted'>Unknown</span>");
    setText('id-dob',        id.dob        ? id.dob.substring(0, 10)        : "N/A");
    setText('id-ssn',        "XXX-XX-"    + (id.ssn ? id.ssn.slice(-4)     : "XXXX"));
    setText('id-reportdate', id.report_date ? id.report_date.substring(0, 10) : "N/A");

    // Bureau Model Score
    const score = data.metrics["Bureau Model Score"];
    const scoreEl = document.getElementById('model-score');
    if (scoreEl) {
        scoreEl.textContent = (score !== null && score !== undefined) ? score : "N/A";
        scoreEl.style.color = score > 0 ? 'var(--success)' : 'var(--text-muted)';
    }

    // ── REVIEW TIER ──────────────────────────────────────────
    const tier = data.tier || "";
    const tierColor =
        tier === "Standard Review" ? "var(--success)" :
        tier === "Enhanced Review" ? "var(--warning)" :
        tier === "Detailed Review" ? "var(--danger)"  : "var(--text-muted)";
    const tierBadgeClass =
        tier === "Standard Review" ? "success" :
        tier === "Enhanced Review" ? "warning" : "danger";

    const reviewTierVal = document.getElementById('review-tier-val');
    if (reviewTierVal) {
        reviewTierVal.textContent = tier || "—";
        reviewTierVal.style.color = tierColor;
    }

    const tierBadgeEl = document.getElementById('tier-badge');
    if (tierBadgeEl) {
        tierBadgeEl.textContent = tier || "—";
        tierBadgeEl.className   = `badge ${tierBadgeClass}`;
    }

    setText('tier-guidance-text', data.tier_guidance || "");

    // ── RISK PILLARS ──────────────────────────────────────────
    const pillarsGrid = document.getElementById('pillars-grid');
    if (pillarsGrid) {
        pillarsGrid.innerHTML = '';
        (data.risk_pillars || []).forEach(p => {
            const riskClass   =
                p.risk === "Elevated" ? "pillar-risk-elevated" :
                p.risk === "Moderate" ? "pillar-risk-moderate" :
                p.risk === "Low"      ? "pillar-risk-low"      :
                                        "pillar-risk-unknown";
            const badgeClass  =
                p.risk === "Elevated" ? "danger"  :
                p.risk === "Moderate" ? "warning" :
                p.risk === "Low"      ? "success" : "neutral";
            const notesHtml = (p.notes || []).map(n => `<li>${n}</li>`).join('');
            pillarsGrid.innerHTML += `
                <div class="pillar-box ${riskClass}">
                    <div class="pillar-header">
                        <span class="pillar-name">${p.pillar}</span>
                        <span class="badge ${badgeClass}">${p.risk}</span>
                    </div>
                    <ul class="pillar-notes">${notesHtml}</ul>
                </div>`;
        });
    }

    // ── ADDRESS FLAG ──────────────────────────────────────────
    const addrElem = document.getElementById('id-address-flag');
    if (addrElem) {
        if (m["Address Discrepancy Flag"]) {
            addrElem.textContent = "Discrepancy Detected";
            addrElem.className   = "id-val text-danger";
        } else {
            addrElem.textContent = "Verified Match";
            addrElem.className   = "id-val text-success";
        }
    }

    // ── CANONICAL DECISIONING METRICS ────────────────────────
    const fmtPct = v => (v == null ? "—" : v.toFixed(2) + "%");
    const fmtCur = v => (v == null ? "—" : formatCurrency(v));
    const fmtInt = v => (v == null ? "—" : Math.round(v));

    setText('canon-bal-active',   fmtCur(m["Total Balance on Open Trades"]));
    setText('canon-bal-revolve',  fmtCur(m["Revolving Balance"]));
    setText('canon-lim-revolve',  fmtCur(m["Revolving Credit Limit"]));
    setText('canon-util-rev3',    fmtPct(m["Revolving Utilization 3m (%)"]));
    setText('canon-util-rev6',    fmtPct(m["Revolving Utilization 6m (%)"]));
    setText('canon-util-bc3',     fmtPct(m["Bankcard Utilization 3m (%)"]));
    setText('canon-util-bc6',     fmtPct(m["Bankcard Utilization 6m (%)"]));
    setText('canon-util-maxrev3', fmtPct(m["Max Revolving Utilization 3m (%)"]));
    setText('canon-past-due',     fmtInt(m["Trades Past Due Ever"]));
    setText('canon-major-derog',  fmtInt(m["Major Derogatory Trades Ever"]));
    setText('canon-pmts-made',    fmtInt(m["Trades with Payment Made Ever"]));
    setText('canon-mos-file',     fmtInt(m["Months on Credit File"]));
    setText('canon-oldest-trade', fmtInt(m["Oldest Trade Age (Months)"]));
    setText('canon-avg-age',      fmtInt(m["Average Trade Age (Months)"]));

    // ── BALANCE & CREDIT OVERVIEW ────────────────────────────
    setText('metric-total-balance',  formatCurrency(m["Total Balance (All Accounts)"]));
    setText('metric-credit-limit',   formatCurrency(m["Total Credit Limit"]));
    setText('metric-high-credit',    formatCurrency(m["Total High Credit (Peak/Loans)"]));
    setText('metric-unused-credit',  formatCurrency(m["Unused Credit"]));

    // ── DEBT & REPAYMENT ──────────────────────────────────────
    setText('metric-debt',           formatCurrency(m["Total Active Debt"]));
    setText('metric-avg-obligation', formatCurrency(m["Average Monthly Obligation"]));
    setText('metric-actual-pmt',     formatCurrency(m["Total Actual Payments Made"]));
    setText('metric-sched-pmt',      formatCurrency(m["Total Scheduled Monthly Payments"]));

    // DSCR with colour coding
    const dscr     = m["Debt Service Coverage Ratio"] || 0;
    const dscrElem = document.getElementById('metric-dscr');
    if (dscrElem) {
        dscrElem.textContent = dscr.toFixed(2);
        dscrElem.className   = dscr < 1.0 ? 'm-val text-danger' : 'm-val';
    }

    // Utilisation bar
    const util = m["Credit Utilisation (%)"] || 0;
    setText('metric-util-val', util.toFixed(2) + "%");
    const utilFill = document.getElementById('util-fill');
    if (utilFill) {
        setTimeout(() => {
            utilFill.style.width = Math.min(util, 100) + "%";
            utilFill.style.backgroundColor =
                util >= 50 ? 'var(--danger)' :
                util >= 30 ? 'var(--warning)' :
                             'var(--success)';
        }, 100);
    }

    // ── ACCOUNT MIX ─────────────────────────────────────────
    setText('metric-total-accounts',   m["Total Accounts"]   || 0);
    setText('metric-active-accounts',  m["Active Accounts"]  || 0);
    setText('metric-closed-accounts',  m["Closed Accounts"]  || 0);
    setText('metric-adverse-count',    m["Adverse Accounts"] || 0);
    setText('metric-mortgage',         m["Mortgage Accounts"]       || 0);
    setText('metric-installment',      m["Installment Accounts"]    || 0);
    setText('metric-revolving',        m["Revolving Accounts"]      || 0);
    setText('metric-paying-agreed',    m["Accounts Paying as Agreed"] || 0);

    // Update account mix badge on table
    setText('account-mix-badge',
        `${m["Active Accounts"] || 0} Active / ${m["Total Accounts"] || 0} Total`);

    // ── CREDIT HISTORY ────────────────────────────────────────
    setText('metric-age',            (m["Credit Age (Months)"] || 0) + " mos");
    setText('metric-max-history',    (m["Max Months of History"] || 0) + " mos");
    setText('metric-oldest-account', m["Oldest Account Opened"] || "—");
    setText('metric-newest-account', m["Newest Account Opened"] || "—");

    // ── ADVERSE ACCOUNTS ──────────────────────────────────────
    const nAdverse = m["Adverse Accounts (Bounced equiv)"] || 0;
    const adBadge  = document.getElementById('adverse-badge');
    if (adBadge) {
        adBadge.textContent = nAdverse + " Adverse";
        adBadge.className   = nAdverse > 0 ? 'badge danger' : 'badge success';
    }

    const advList = document.getElementById('adverse-list');
    if (advList) {
        advList.innerHTML = '';
        if (!data.adverse_accounts || data.adverse_accounts.length === 0) {
            advList.innerHTML = '<div style="color:var(--text-muted); font-size: 0.9rem; font-style: italic;">No derogatory/adverse marks found.</div>';
        } else {
            data.adverse_accounts.forEach(a => {
                advList.innerHTML += `
                 <div class="adverse-item">
                    <div class="title">${a.name}</div>
                    <div class="desc">${a["Bounce Category"]}</div>
                 </div>`;
            });
        }
    }

    // ── INQUIRY VELOCITY ──────────────────────────────────────
    const inq = data.inquiry_velocity || {};
    setText('stat-inq-total', inq.total || 0);
    setText('stat-inq-3m',    m["Inquiries (Last 3 Months)"]  || 0);
    setText('stat-inq-6m',    m["Inquiries (Last 6 Months)"]  || 0);
    setText('stat-inq-12m',   m["Inquiries (Last 12 Months)"] || 0);

    const clBadge = document.getElementById('cluster-badge');
    if (clBadge) {
        if (inq.clustered_inquiry_dates && inq.clustered_inquiry_dates.length > 0) {
            clBadge.textContent = inq.clustered_inquiry_dates.length + " Date Clusters!";
            clBadge.className   = "badge danger";
        } else {
            clBadge.textContent = "Normal Rate";
            clBadge.className   = "badge success";
        }
    }

    // ── LENDER SUMMARY ────────────────────────────────────────
    setText('stat-lenders',      data.n_distinct_lenders || 0);
    setText('stat-missing-pmts', (data.lenders_missing_payments || []).length);

    const missEl = document.getElementById('missing-payments-list');
    if (missEl) {
        missEl.innerHTML = '';
        (data.lenders_missing_payments || []).slice(0, 6).forEach(lender => {
            missEl.innerHTML += `<span class="missing-chip">${lender}</span>`;
        });
        if ((data.lenders_missing_payments || []).length > 6) {
            missEl.innerHTML += `<span style="font-size:0.78rem; color:var(--text-muted);">+${data.lenders_missing_payments.length - 6} more</span>`;
        }
    }

    // ── ACTIVE TRADELINES TABLE ───────────────────────────────
    const tbody = document.getElementById('accounts-tbody');
    if (tbody) {
        tbody.innerHTML = '';
        if (!data.monthly_balance_report || data.monthly_balance_report.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 2rem;">No active tradelines found</td></tr>';
        } else {
            data.monthly_balance_report.slice(0, 15).forEach(acc => {
                const statusBadge = acc.Status === "Pays as agreed"
                    ? '<span class="badge success">OK</span>'
                    : '<span class="badge danger">Issue</span>';

                const limitAmt = acc["Credit Limit"] > 0
                    ? formatCurrency(acc["Credit Limit"])
                    : formatCurrency(acc["High Credit / Loan Amount"]);

                tbody.innerHTML += `
                    <tr>
                        <td>
                            <div style="font-weight:500;">${acc.Lender}</div>
                            <div style="font-size:0.75rem; color:var(--text-muted)">${acc["Account Type"]} | Port: ${acc.Portfolio}</div>
                        </td>
                        <td style="font-family:'Outfit'; font-weight:500;">${formatCurrency(acc["Current Balance"])}</td>
                        <td>${limitAmt}</td>
                        <td>${(acc["Utilisation (%)"] || 0).toFixed(1)}%</td>
                        <td>${formatCurrency(acc["Scheduled Payment"])}</td>
                        <td>${statusBadge}</td>
                    </tr>`;
            });
        }
    }
}

// ─── Excel Export ─────────────────────────────────────────────
function downloadExcel() {
    if (!_lastData) return;

    const m  = _lastData.metrics  || {};
    const id = _lastData.identity || {};
    const wb = XLSX.utils.book_new();

    // ── Sheet 1: Summary ──────────────────────────────────────
    const summaryRows = [
        ["Field", "Value"],
        ["Full Name",              id.full_name || ""],
        ["Date of Birth",          id.dob        ? id.dob.substring(0,10) : ""],
        ["SSN (last 4)",           id.ssn        ? "XXX-XX-" + id.ssn.slice(-4) : ""],
        ["Report Date",            id.report_date ? id.report_date.substring(0,10) : ""],
        ["Address Discrepancy",    m["Address Discrepancy Flag"] ? "Yes" : "No"],
        ["Bureau Model Score",     m["Bureau Model Score"] ?? "N/A"],
        [],
        ["Review Tier",            _lastData.tier || ""],
        ["Tier Guidance",          _lastData.tier_guidance || ""],
        [],
        ["Distinct Active Lenders",         _lastData.n_distinct_lenders || 0],
        ["Lenders Missing Sched. Payments", (_lastData.lenders_missing_payments || []).length],
    ];
    const wsSum = XLSX.utils.aoa_to_sheet(summaryRows);
    wsSum['!cols'] = [{wch: 36}, {wch: 28}];
    XLSX.utils.book_append_sheet(wb, wsSum, "Summary");

    // ── Sheet 2: Key Metrics (raw tradeline) ─────────────────
    const metricOrder = [
        "Total Balance (All Accounts)",
        "Total Credit Limit",
        "Total High Credit (Peak/Loans)",
        "Unused Credit",
        "Credit Utilisation (%)",
        "Total Active Debt",
        "Total Scheduled Monthly Payments",
        "Total Actual Payments Made",
        "Debt Service Coverage Ratio",
        "Average Monthly Obligation",
        "Total Accounts",
        "Active Accounts",
        "Closed Accounts",
        "Mortgage Accounts",
        "Installment Accounts",
        "Revolving Accounts",
        "Accounts Paying as Agreed",
        "Adverse Accounts",
        "Credit Age (Months)",
        "Oldest Account Opened",
        "Newest Account Opened",
        "Max Months of History",
        "Total Inquiries",
        "Inquiries (Last 3 Months)",
        "Inquiries (Last 6 Months)",
        "Inquiries (Last 12 Months)",
        "Adverse Accounts (Bounced equiv)",
        "Address Discrepancy Flag",
        "Bureau Model Score",
    ];
    const metricRows = [["Metric", "Value"],
        ...metricOrder.map(key => [key, m[key] !== undefined ? m[key] : "N/A"])
    ];
    const wsMetrics = XLSX.utils.aoa_to_sheet(metricRows);
    wsMetrics['!cols'] = [{wch: 42}, {wch: 20}];
    XLSX.utils.book_append_sheet(wb, wsMetrics, "Key Metrics");

    // ── Sheet 3: Canonical Decisioning Metrics ───────────────
    const canonOrder = [
        ["Total Accounts (Equifax attr 4140)",        "Total Accounts"],
        ["Open Trades (Equifax attr 4173)",           "Active Accounts"],
        ["Closed/Settled Trades",                     "Closed Accounts"],
        ["Total Balance on Open Trades (attr 4749)",  "Total Balance on Open Trades"],
        ["Revolving Balance (trade-level computed)",  "Revolving Balance"],
        ["Revolving Credit Limit (trade-level)",      "Revolving Credit Limit"],
        ["Revolving Utilization 3m % (attr 4945)",    "Revolving Utilization 3m (%)"],
        ["Revolving Utilization 6m % (attr 4946)",    "Revolving Utilization 6m (%)"],
        ["Bankcard Utilization 3m % (attr 4932)",     "Bankcard Utilization 3m (%)"],
        ["Bankcard Utilization 6m % (attr 4933)",     "Bankcard Utilization 6m (%)"],
        ["Retail Revolving Util 3m % (attr 4938)",    "Retail Revolving Utilization 3m (%)"],
        ["Max Revolving Util 3m % (attr 5048)",       "Max Revolving Utilization 3m (%)"],
        ["Trades Past Due Ever (attr 4202)",          "Trades Past Due Ever"],
        ["Major Derogatory Trades Ever (attr 3542)",  "Major Derogatory Trades Ever"],
        ["Trades w/ Payment Made Ever (attr 3497)",   "Trades with Payment Made Ever"],
        ["Months on Credit File (attr 5798)",         "Months on Credit File"],
        ["Oldest Trade Age Months (attr 3001)",       "Oldest Trade Age (Months)"],
        ["Average Trade Age Months (attr 3108)",      "Average Trade Age (Months)"],
    ];
    const canonRows = [["Field (Equifax Attribute)", "Value"],
        ...canonOrder.map(([label, key]) => [label, m[key] != null ? m[key] : "N/A"])
    ];
    const wsCanon = XLSX.utils.aoa_to_sheet(canonRows);
    wsCanon['!cols'] = [{wch: 52}, {wch: 18}];
    XLSX.utils.book_append_sheet(wb, wsCanon, "Canonical Metrics");

    // ── Sheet 4: Active Tradelines ────────────────────────────
    const trades = _lastData.monthly_balance_report || [];
    if (trades.length > 0) {
        const headers   = Object.keys(trades[0]);
        const tradeRows = [headers, ...trades.map(t => headers.map(h => t[h] ?? ""))];
        const wsTrades  = XLSX.utils.aoa_to_sheet(tradeRows);
        wsTrades['!cols'] = headers.map(() => ({wch: 22}));
        XLSX.utils.book_append_sheet(wb, wsTrades, "Active Tradelines");
    }

    // ── Sheet 4: Adverse Accounts ─────────────────────────────
    const adverse = _lastData.adverse_accounts || [];
    if (adverse.length > 0) {
        const aHeaders  = Object.keys(adverse[0]);
        const aRows     = [aHeaders, ...adverse.map(a =>
            aHeaders.map(h => { const v = a[h]; return Array.isArray(v) ? v.join(", ") : (v ?? ""); })
        )];
        const wsAdverse = XLSX.utils.aoa_to_sheet(aRows);
        wsAdverse['!cols'] = aHeaders.map(() => ({wch: 26}));
        XLSX.utils.book_append_sheet(wb, wsAdverse, "Adverse Accounts");
    }

    // ── Sheet 5: Lenders Missing Payments ────────────────────
    const missing = _lastData.lenders_missing_payments || [];
    if (missing.length > 0) {
        const mRows    = [["Lender (No Scheduled Payment)"], ...missing.map(l => [l])];
        const wsMissing = XLSX.utils.aoa_to_sheet(mRows);
        wsMissing['!cols'] = [{wch: 40}];
        XLSX.utils.book_append_sheet(wb, wsMissing, "Missing Payments");
    }

    // ── Sheet 6: Risk Assessment ──────────────────────────────
    const pillarsData = _lastData.risk_pillars || [];
    if (pillarsData.length > 0) {
        const pRows = [
            ["Pillar", "Risk Level", "Risk Score", "Notes"],
            ...pillarsData.map(p => [
                p.pillar,
                p.risk,
                p.risk_score,
                (p.notes || []).join(" | ")
            ])
        ];
        const wsPillars = XLSX.utils.aoa_to_sheet(pRows);
        wsPillars['!cols'] = [{wch: 22}, {wch: 12}, {wch: 12}, {wch: 90}];
        XLSX.utils.book_append_sheet(wb, wsPillars, "Risk Assessment");
    }

    // ── Filename: ApplicantName_YYYY-MM-DD.xlsx ───────────────
    const name = (id.full_name || "Bureau_Report").replace(/\s+/g, "_");
    const date = new Date().toISOString().slice(0, 10);
    XLSX.writeFile(wb, `${name}_${date}.xlsx`);
}
