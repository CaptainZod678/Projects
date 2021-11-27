package com.example.accountapp.model;

import java.util.ArrayList;
import java.util.List;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.OneToMany;
import javax.persistence.Table;

import lombok.Data;

@Data
@Entity
@Table(name = "accounts")
public class Account {
	
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY )
	private long id;
	
	@Column(name = "account_name")
	private String accountName;
	
	@Column(name = "profile_id")
	private String profileId;
	
	@Column(name = "total_balance")
	private int totalBalance;
	
	@OneToMany(targetEntity = Holding.class ,cascade = CascadeType.ALL)
	@JoinColumn(name = "ac_hid", referencedColumnName = "id")
	List<Holding> holding = new ArrayList<>();
   


}
