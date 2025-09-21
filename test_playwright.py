from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page()
        page.goto("https://example.com")
        print("Title:", page.title())
        b.close()

if __name__ == "__main__":
    main()
