<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Uber Trip Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg: #f1f2f6;
            --text: #2c3e50;
            --card: #ffffff;
            --primary: #0984e3;
            --nav: #dcdde1;
        }
        body.dark {
            --bg: #1e272e;
            --text: #f5f6fa;
            --primary: #00a8ff;
            --nav: #353b48;
        }
        body {
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            transition: background 0.3s, color 0.3s;
        }
        header {
            background: var(--text);
            color: var(--card);
            padding: 20px;
            text-align: center;
            font-size: 2em;
            position: relative;
        }
        .theme-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 1.5em;
            cursor: pointer;
        }
        nav {
            display: flex;
            justify-content: center;
            background: var(--nav);
            padding: 10px 0;
        }
        nav ul {
            list-style: none;
            display: flex;
            padding: 0;
            margin: 0;
        }
        nav li {
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 6px;
            margin: 0 5px;
            background: #dfe6e9;
            transition: 0.2s;
        }
        nav li.active,
        nav li:hover {
            background: var(--primary);
            color: black;
        }
        .tab-content {
            display: none;
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .tab-content.active {
            display: block;
        }
        .actions {
            text-align: center;
            margin-top: 20px;
        }
        .btn {
            background: #00cec9;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        .btn:hover {
            background: var(--primary);
        }
        footer {
            text-align: center;
            padding: 20px;
            background: var(--text);
            color: var(--card);
            margin-top: 40px;
        }
        .dark .tab-content {
            background: #2f3640;
            color: #f5f6fa;
        }
        .dark header,
        .dark footer {
            background: #1e272e;
        }
        .dark nav {
            background: #2d3436;
        }
        .dark nav li {
            background: #636e72;
        }
        .dark nav li.active,
        .dark nav li:hover {
            background: #00cec9;
            color: #1e272e;
        }
    </style>
</head>
<body>
    <header>
        📊 Uber Trip Forecasting Dashboard
        <div class="theme-toggle" id="toggle-theme">🌓</div>
    </header>
    <nav>
        <ul>
            {tab_headers}
        </ul>
    </nav>
    {tab_contents}
    <div class="actions">
        <form action="/export/pdf">
            <button class="btn" type="submit">📄 Export All Plots to PDF</button>
        </form>
    </div>
    <footer>Built by Jayanth Chennoju | Tools: FastAPI, XGBoost, Plotly, SHAP, Render</footer>

    <script>
        const toggleIcon = document.getElementById('toggle-theme');

        const setTheme = (dark) => {
            document.body.classList.toggle('dark', dark);
            localStorage.setItem('theme', dark ? 'dark' : 'light');

            const plotlyFrames = document.querySelectorAll("iframe");
            plotlyFrames.forEach(iframe => {
                iframe.contentWindow?.Plotly?.relayout?.(
                    iframe.contentWindow.document.querySelector("div.js-plotly-plot"),
                    { template: dark ? "plotly_dark" : "plotly_white" }
                );
            });
        };

        const savedTheme = localStorage.getItem('theme') === 'dark';
        setTheme(savedTheme);

        toggleIcon.addEventListener('click', () => {
            const darkMode = !document.body.classList.contains('dark');
            setTheme(darkMode);
        });

        document.querySelectorAll('nav li').forEach((tab, index) => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('nav li').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById("tab" + index).classList.add('active');
            });
        });
    </script>
</body>
</html>
