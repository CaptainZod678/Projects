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
@Table(name = "holdings")
public class Holding {

   @Id
   @GeneratedValue(strategy = GenerationType.IDENTITY)
   private long hid;
   
   @Column(name = "hname")
   private String holdingName;
   
   @Column(name = "hbalance")
   private long balance;
   
   @Column(name = "htype")
   private String holdingType ;
   
   @Column(name = "hquantity")
   private long quantity;
   
	
}
