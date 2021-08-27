library(data.table)
library(ggplot2)

orig_json = jsonlite::fromJSON("./bench_data/origin_develop_inv_phi_test.json")
new_json = jsonlite::fromJSON("./bench_data/spinkney_faster_inv_phi_inv_phi_test.json")

context_lst = orig_json[[1]]
orig_dt = as.data.table(orig_json[[2]])
new_dt = as.data.table(new_json[[2]])
new_dt[, version := "PR"]
orig_dt[, version := "Develop"]
raw_full_dt = rbind(new_dt, orig_dt)
get_sizes = function(x) {
  as.numeric(gsub(".*/([0-9]+)/.*", "\\1", x))
}
raw_full_dt[, size := get_sizes(name)]
setkey(raw_full_dt, size, version)
sub_full_dt = raw_full_dt[, .(real_time, size, version)]

full_cast_dt = dcast(sub_full_dt, size ~ version, value.var = "real_time")

bench_name = gsub("./benches/", "", context_lst$executable)
ggplot(full_cast_dt, aes(x = size)) +
  geom_line(aes(y = log(Develop)), color = "red") +
  geom_line(aes(y = log(PR)), color = "blue") +
  scale_x_log10() +
  ggtitle(paste0("Log-Log Sizes Vs. Times for Forward and Reverse Pass of ", bench_name),
          "Red: Current -- Blue: PR") +
  theme_bw(base_size = 18) + xlab("Sizes") + ylab("")

ggplot(full_cast_dt, aes(x = size)) +
  geom_line(aes(y = (Develop / PR) - 1)) +
  geom_hline(yintercept = 0) +
  scale_x_log10() +
  ggtitle(paste0("(Current / PR) - 1 for Reverse Pass of ", bench_name),
    "0 is the base while anything greater is X magnitude speed increase") +
  theme_bw(base_size = 18) + xlab("Sizes") + ylab("")


