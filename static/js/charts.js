/**
 * Interactive charts for RLVR presentation.
 * Uses Plotly.js — data approximated from Tulu 3 paper Figure 19.
 *
 * Color tokens aligned with theme.css design system.
 */

const COLORS = {
    accent: '#4d8eff',
    green: '#34d399',
    red: '#f87171',
    orange: '#fbbf24',
    purple: '#a78bfa',
    text: '#9898a6',
    textMuted: '#65657a',
    bg: '#141418',
    grid: 'rgba(255,255,255,0.06)',
    zero: 'rgba(255,255,255,0.08)'
};

const LAYOUT_BASE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, -apple-system, sans-serif', color: COLORS.text, size: 13 },
    margin: { l: 55, r: 20, t: 40, b: 42 },
    xaxis: {
        gridcolor: COLORS.grid,
        zerolinecolor: COLORS.zero,
        title: { text: 'Episodes (thousands)', standoff: 10, font: { size: 12, color: COLORS.textMuted } },
        tickfont: { size: 11 }
    },
    yaxis: {
        gridcolor: COLORS.grid,
        zerolinecolor: COLORS.zero,
        tickfont: { size: 11 }
    },
    legend: {
        bgcolor: 'rgba(0,0,0,0)',
        font: { size: 11, color: COLORS.text }
    }
};

function initCharts() {
    const rewardEl = document.getElementById('chart-rewards');
    if (rewardEl) {
        renderRewardChart(rewardEl);
    }

    const progressEl = document.getElementById('chart-llm-progress');
    if (progressEl) {
        renderLLMProgressChart(progressEl);
    }

    const errorRateEl = document.getElementById('chart-error-rate');
    if (errorRateEl) {
        renderErrorRateChart(errorRateEl);
    }

    const expertEl = document.getElementById('chart-expert-domains');
    if (expertEl) {
        renderExpertDomainsChart(expertEl);
    }
}

function renderRewardChart(el) {
    // Approximated from Figure 19 — GSM8K rewards for different beta values
    const episodes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

    const traces = [
        {
            x: episodes,
            y: [7.0, 7.8, 8.2, 8.5, 8.7, 8.8, 8.9, 8.9, 9.0, 9.0, 9.0],
            name: '\u03B2 = 0.01 (low penalty)',
            line: { color: COLORS.red, width: 2, shape: 'spline' },
            mode: 'lines+markers',
            marker: { size: 4, color: COLORS.red }
        },
        {
            x: episodes,
            y: [7.0, 7.6, 8.0, 8.3, 8.4, 8.5, 8.6, 8.6, 8.7, 8.7, 8.7],
            name: '\u03B2 = 0.03',
            line: { color: COLORS.orange, width: 2, shape: 'spline' },
            mode: 'lines+markers',
            marker: { size: 4, color: COLORS.orange }
        },
        {
            x: episodes,
            y: [7.0, 7.5, 7.8, 8.0, 8.2, 8.3, 8.3, 8.4, 8.4, 8.5, 8.5],
            name: '\u03B2 = 0.05 (sweet spot)',
            line: { color: COLORS.green, width: 2.5, shape: 'spline' },
            mode: 'lines+markers',
            marker: { size: 5, color: COLORS.green }
        },
        {
            x: episodes,
            y: [7.0, 7.3, 7.5, 7.6, 7.7, 7.8, 7.8, 7.9, 7.9, 7.9, 8.0],
            name: '\u03B2 = 0.1 (high penalty)',
            line: { color: COLORS.purple, width: 2, shape: 'spline' },
            mode: 'lines+markers',
            marker: { size: 4, color: COLORS.purple }
        }
    ];

    const layout = {
        ...LAYOUT_BASE,
        title: {
            text: 'RLVR Training: Verifiable Rewards on GSM8K',
            font: { size: 14, color: COLORS.text, weight: 600 },
            x: 0.01,
            xanchor: 'left'
        },
        xaxis: {
            ...LAYOUT_BASE.xaxis,
            title: { text: 'Episodes (thousands)', standoff: 10, font: { size: 12, color: COLORS.textMuted } }
        },
        yaxis: {
            ...LAYOUT_BASE.yaxis,
            title: { text: 'Verifiable Rewards', standoff: 10, font: { size: 12, color: COLORS.textMuted } },
            range: [6.5, 9.5]
        },
        legend: { ...LAYOUT_BASE.legend, x: 0.02, y: 0.98 },
        annotations: [{
            x: 50, y: 8.3,
            text: 'sweet spot',
            showarrow: true,
            arrowhead: 0,
            arrowwidth: 1.5,
            arrowcolor: COLORS.green,
            font: { color: COLORS.green, size: 11 },
            ax: 40, ay: -30,
            bgcolor: 'rgba(20, 20, 24, 0.8)',
            borderpad: 4,
            bordercolor: 'rgba(52, 211, 153, 0.3)',
            borderwidth: 1
        }]
    };

    Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
}

function renderLLMProgressChart(el) {
    // Model milestones — approximate benchmark composite scores (MMLU-like)
    // Scores are illustrative, not exact, to show the trend
    const models = [
        { name: 'GPT-3', date: '2020-06-01', score: 43 },
        { name: 'PaLM', date: '2022-04-01', score: 56 },
        { name: 'ChatGPT', date: '2022-11-30', score: 60 },
        { name: 'GPT-4', date: '2023-03-14', score: 75 },
        { name: 'Claude 3', date: '2024-03-04', score: 78 },
        { name: 'Gemini 1.5', date: '2024-05-01', score: 79 },
        { name: 'GPT-4o', date: '2024-05-13', score: 81 },
        { name: 'Claude 3.5\nSonnet', date: '2024-06-20', score: 82 },
        { name: 'o1', date: '2024-09-12', score: 86 },
        { name: 'Claude 4.5\nSonnet', date: '2025-02-24', score: 88 },
        { name: 'Claude\nOpus 4.6', date: '2025-12-01', score: 91 },
        { name: 'Mythos\nPreview', date: '2026-04-07', score: 95 },
    ];

    // Historical data points
    const histTrace = {
        x: models.map(m => m.date),
        y: models.map(m => m.score),
        text: models.map(m => m.name),
        mode: 'lines+markers+text',
        type: 'scatter',
        line: { color: 'rgba(255,255,255,0.25)', width: 1.5, shape: 'spline' },
        textposition: models.map((m, i) => {
            // Alternate text position to avoid overlap
            if (m.name.includes('PaLM') || m.name.includes('Gemini') || m.name.includes('o1')) return 'bottom center';
            if (i % 2 === 0) return 'top center';
            return 'bottom center';
        }),
        textfont: { size: 9.5, color: COLORS.text, family: 'Inter, sans-serif' },
        marker: {
            size: models.map(m => m.name.includes('Mythos') ? 12 : 8),
            color: models.map(m => {
                if (m.name.includes('Claude') || m.name.includes('Mythos')) return COLORS.accent;
                return COLORS.text;
            }),
            symbol: models.map(m => {
                if (m.name.includes('Claude') || m.name.includes('Mythos')) return 'diamond';
                return 'circle';
            }),
            line: { width: 1, color: 'rgba(255,255,255,0.3)' }
        },
        hoverinfo: 'text+y',
        showlegend: false
    };

    // Projection trajectory (2026-2028) with confidence band
    const projDates = ['2026-04-07', '2026-10-01', '2027-04-01', '2027-10-01', '2028-04-01'];
    const projMid = [95, 96.5, 97.5, 98, 98.3];
    const projHigh = [95, 98, 99.5, 100, 100];
    const projLow = [95, 95, 95.5, 95.5, 96];

    const projLine = {
        x: projDates,
        y: projMid,
        mode: 'lines',
        line: { color: COLORS.accent, width: 2, dash: 'dot' },
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Upper bound (invisible, for fill)
    const projUpper = {
        x: projDates,
        y: projHigh,
        mode: 'lines',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Lower bound with fill to upper
    const projLower = {
        x: projDates,
        y: projLow,
        mode: 'lines',
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(77, 142, 255, 0.08)',
        showlegend: false,
        hoverinfo: 'skip'
    };

    const layout = {
        ...LAYOUT_BASE,
        title: {
            text: 'LLM Capability Over Time',
            font: { size: 14, color: COLORS.text, weight: 600 },
            x: 0.01,
            xanchor: 'left'
        },
        xaxis: {
            ...LAYOUT_BASE.xaxis,
            type: 'date',
            title: { text: '', standoff: 10 },
            tickformat: '%Y',
            tickfont: { size: 11, color: COLORS.textMuted },
            range: ['2019-10-01', '2028-07-01'],
            dtick: 'M12'
        },
        yaxis: {
            ...LAYOUT_BASE.yaxis,
            title: { text: 'Benchmark Composite', standoff: 10, font: { size: 12, color: COLORS.textMuted } },
            range: [35, 105],
            ticksuffix: '%'
        },
        margin: { l: 60, r: 30, t: 40, b: 42 },
        annotations: [{
            x: '2027-04-01', y: 97.5,
            text: 'projection',
            showarrow: false,
            font: { color: COLORS.textMuted, size: 10 },
            bgcolor: 'rgba(20, 20, 24, 0.8)',
            borderpad: 3
        }],
        shapes: [{
            type: 'line',
            x0: '2026-04-07', x1: '2026-04-07',
            y0: 35, y1: 100,
            line: { color: 'rgba(255,255,255,0.08)', width: 1, dash: 'dot' }
        }]
    };

    // Order matters: upper bound first, then lower (for fill reference)
    Plotly.newPlot(el, [projUpper, projLower, projLine, histTrace], layout, { responsive: true, displayModeBar: false });
}

function renderErrorRateChart(el) {
    // Same model data as renderLLMProgressChart, but y = 100 - score (error rate)
    const models = [
        { name: 'GPT-3', date: '2020-06-01', score: 43 },
        { name: 'PaLM', date: '2022-04-01', score: 56 },
        { name: 'ChatGPT', date: '2022-11-30', score: 60 },
        { name: 'GPT-4', date: '2023-03-14', score: 75 },
        { name: 'Claude 3', date: '2024-03-04', score: 78 },
        { name: 'Gemini 1.5', date: '2024-05-01', score: 79 },
        { name: 'GPT-4o', date: '2024-05-13', score: 81 },
        { name: 'Claude 3.5\nSonnet', date: '2024-06-20', score: 82 },
        { name: 'o1', date: '2024-09-12', score: 86 },
        { name: 'Claude 4.5\nSonnet', date: '2025-02-24', score: 88 },
        { name: 'Claude\nOpus 4.6', date: '2025-12-01', score: 91 },
        { name: 'Mythos\nPreview', date: '2026-04-07', score: 95 },
    ];

    // Historical data trace — error rate = 100 - score
    const histTrace = {
        x: models.map(m => m.date),
        y: models.map(m => 100 - m.score),
        text: models.map(m => m.name),
        mode: 'lines+markers+text',
        type: 'scatter',
        line: { color: 'rgba(255,255,255,0.25)', width: 1.5, shape: 'spline' },
        textposition: models.map((m, i) => {
            // Inverted from slide 2: low-error models (bottom) get labels above
            if (m.name.includes('PaLM') || m.name.includes('Gemini') || m.name.includes('o1')) return 'top center';
            if (i % 2 === 0) return 'bottom center';
            return 'top center';
        }),
        textfont: { size: 9.5, color: COLORS.text, family: 'Inter, sans-serif' },
        marker: {
            size: models.map(m => m.name.includes('Mythos') ? 12 : 8),
            color: models.map(m => {
                if (m.name.includes('Claude') || m.name.includes('Mythos')) return COLORS.accent;
                return COLORS.text;
            }),
            symbol: models.map(m => {
                if (m.name.includes('Claude') || m.name.includes('Mythos')) return 'diamond';
                return 'circle';
            }),
            line: { width: 1, color: 'rgba(255,255,255,0.3)' }
        },
        hovertemplate: '%{text}<br>Error rate: %{y:.1f}%<extra></extra>',
        showlegend: false
    };

    // Exponential decay projection (2026-04 to 2028-04)
    // error(t) = baseError * exp(-lambda * t), t in years from anchor
    const baseError = 5; // Mythos error rate (100 - 95)
    const lambdaFast = Math.LN2 / 0.6;  // optimistic: halves every 0.6 years
    const lambdaSlow = Math.LN2 / 2.0;  // pessimistic: halves every 2.0 years
    const lambda     = Math.LN2 / 0.86; // mid: ~1% error at 2028

    const projDates = ['2026-04-07', '2026-10-07', '2027-04-07', '2027-10-07', '2028-04-07'];
    const projT     = [0, 0.5, 1.0, 1.5, 2.0]; // years from anchor

    const projMid  = projT.map(t => baseError * Math.exp(-lambda * t));
    const projHigh = projT.map(t => baseError * Math.exp(-lambdaSlow * t)); // pessimistic (higher error)
    const projLow  = projT.map(t => baseError * Math.exp(-lambdaFast * t)); // optimistic (lower error)

    const projLine = {
        x: projDates,
        y: projMid,
        mode: 'lines',
        line: { color: COLORS.accent, width: 2, dash: 'dot' },
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Upper bound (pessimistic — higher error values)
    const projUpper = {
        x: projDates,
        y: projHigh,
        mode: 'lines',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip'
    };

    // Lower bound with fill to upper
    const projLower = {
        x: projDates,
        y: projLow,
        mode: 'lines',
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(77, 142, 255, 0.08)',
        showlegend: false,
        hoverinfo: 'skip'
    };

    const layout = {
        ...LAYOUT_BASE,
        title: {
            text: 'Error Rate Over Time (Log Scale)',
            font: { size: 14, color: COLORS.text, weight: 600 },
            x: 0.01,
            xanchor: 'left'
        },
        xaxis: {
            ...LAYOUT_BASE.xaxis,
            type: 'date',
            title: { text: '', standoff: 10 },
            tickformat: '%Y',
            tickfont: { size: 11, color: COLORS.textMuted },
            range: ['2019-10-01', '2028-07-01'],
            dtick: 'M12'
        },
        yaxis: {
            ...LAYOUT_BASE.yaxis,
            type: 'log',
            title: { text: 'Error Rate (100 \u2212 score)', standoff: 10, font: { size: 12, color: COLORS.textMuted } },
            range: [Math.log10(0.2), Math.log10(70)],
            ticksuffix: '%',
            tickvals: [0.5, 1, 2, 5, 10, 20, 50],
            ticktext: ['0.5', '1', '2', '5', '10', '20', '50']
        },
        margin: { l: 60, r: 30, t: 40, b: 42 },
        annotations: [
            {
                x: '2027-06-01', y: Math.log10(2.0),
                text: 'projection',
                showarrow: false,
                font: { color: COLORS.textMuted, size: 10 },
                bgcolor: 'rgba(20, 20, 24, 0.8)',
                borderpad: 3
            },
            {
                x: '2020-01-01', y: Math.log10(55),
                text: '\u2190 higher error',
                showarrow: false,
                font: { color: COLORS.textMuted, size: 9 }
            },
            {
                x: '2020-01-01', y: Math.log10(0.4),
                text: '\u2190 lower error',
                showarrow: false,
                font: { color: COLORS.textMuted, size: 9 }
            }
        ],
        shapes: [{
            type: 'line',
            x0: '2026-04-07', x1: '2026-04-07',
            y0: 0.2, y1: 70,
            line: { color: 'rgba(255,255,255,0.08)', width: 1, dash: 'dot' }
        }]
    };

    // Order matters: upper bound first, then lower (for fill reference)
    Plotly.newPlot(el, [projUpper, projLower, projLine, histTrace], layout, { responsive: true, displayModeBar: false });
}

function renderExpertDomainsChart(el) {
    const categories = ['Math', 'Chemistry', 'Coding', 'Economics', 'Medicine'];

    // Expert group scores: each excels in their domain, average elsewhere
    const expertGroups = [
        { name: 'Mathematicians', scores: [95, 50, 55, 45, 40], color: COLORS.red },
        { name: 'Chemists', scores: [45, 92, 40, 50, 55], color: COLORS.orange },
        { name: 'Software Devs', scores: [60, 40, 94, 45, 35], color: COLORS.green },
        { name: 'Economists', scores: [55, 45, 50, 93, 40], color: COLORS.purple },
        { name: 'Doctors', scores: [40, 60, 35, 45, 96], color: COLORS.accent },
    ];

    // RLVR combined model: excels across all
    const rlvrScores = [92, 88, 91, 89, 90];

    // Start with just the expert group traces (faded)
    const traces = expertGroups.map(group => ({
        type: 'scatterpolar',
        r: [...group.scores, group.scores[0]], // close the polygon
        theta: [...categories, categories[0]],
        name: group.name,
        fill: 'toself',
        fillcolor: group.color.replace(')', ', 0.06)').replace('rgb', 'rgba'),
        line: { color: group.color, width: 1.5 },
        marker: { size: 4 },
        opacity: 0.5
    }));

    // RLVR combined trace — bold and prominent
    traces.push({
        type: 'scatterpolar',
        r: [...rlvrScores, rlvrScores[0]],
        theta: [...categories, categories[0]],
        name: 'RLVR Model',
        fill: 'toself',
        fillcolor: 'rgba(77, 142, 255, 0.12)',
        line: { color: COLORS.accent, width: 3 },
        marker: { size: 7, symbol: 'diamond', color: COLORS.accent },
        opacity: 1
    });

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, -apple-system, sans-serif', color: COLORS.text, size: 12 },
        polar: {
            bgcolor: 'rgba(0,0,0,0)',
            radialaxis: {
                visible: true,
                range: [0, 100],
                tickfont: { size: 9, color: COLORS.textMuted },
                gridcolor: COLORS.grid,
                linecolor: 'rgba(0,0,0,0)',
                ticksuffix: '',
                dtick: 25
            },
            angularaxis: {
                tickfont: { size: 12, color: COLORS.text },
                gridcolor: COLORS.grid,
                linecolor: 'rgba(0,0,0,0)'
            }
        },
        legend: {
            bgcolor: 'rgba(0,0,0,0)',
            font: { size: 10, color: COLORS.text },
            x: 1.05,
            y: 0.5,
            orientation: 'v'
        },
        margin: { l: 60, r: 140, t: 30, b: 30 },
        showlegend: true
    };

    Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });

    // Animation: start with only expert groups visible, then reveal RLVR trace
    // Set initial state: hide RLVR trace
    Plotly.restyle(el, { visible: false }, [traces.length - 1]);

    // After 1.5s, fade in the RLVR trace and dim expert traces
    setTimeout(() => {
        // Show RLVR trace
        Plotly.restyle(el, { visible: true }, [traces.length - 1]);
        // Dim expert traces
        for (let i = 0; i < expertGroups.length; i++) {
            Plotly.restyle(el, { opacity: 0.25 }, [i]);
        }
    }, 1500);
}

// Initialize charts when Reveal.js loads each slide
if (typeof Reveal !== 'undefined') {
    Reveal.on('slidechanged', () => { setTimeout(initCharts, 100); });
    Reveal.on('ready', () => { setTimeout(initCharts, 200); });
} else {
    document.addEventListener('DOMContentLoaded', () => { setTimeout(initCharts, 500); });
}
