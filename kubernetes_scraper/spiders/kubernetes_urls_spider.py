
import scrapy


class KubernetesDocsSpider(scrapy.Spider):
    name = "kubernetes_docs"
    allowed_domains = ["kubernetes.io"]

    start_urls = [
        "https://kubernetes.io/docs/home/",
    ]

    custom_settings = {
        "DEPTH_LIMIT": 0,
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        """Discover and recursively crawl Kubernetes documentation URLs."""

        for href in response.css("a[href]::attr(href)").getall():
            url = response.urljoin(href).split("#")[0].split("?")[0]

            if not (
                url.startswith("https://kubernetes.io/docs/")
                and url.endswith("/")
            ):
                continue

            yield scrapy.Request(
                url=url,
                callback=self.parse,
                dont_filter=False,
            )

            # Export the current page, not every link found on it.
        yield {
            "url": response.url,
            "status": response.status,
        }
