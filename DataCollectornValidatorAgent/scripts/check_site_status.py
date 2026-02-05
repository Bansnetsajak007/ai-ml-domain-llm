from playwright.sync_api import sync_playwright

url = "https://z-lib.sk/"  # Must match BASE_URL in mcp_server.py

def check_site_status() -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=10000)  # 10 sec timeout

            title = page.title()
            browser.close()

            print(f"[✔] {url} is up and working! Title: {title}")
            return "Success"
    except Exception as e:
        print(f"[✖] {url} failed: {e}")
        return "Failed"

# Call the function
check_site_status()
